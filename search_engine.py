import lancedb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import ast
import pandas as pd
import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

class RecipeSearchEngine:
    def __init__(self):
        self.db = lancedb.connect(DB_PATH)
        self.table = self.db.open_table(TABLE_NAME)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize embedding registry for hybrid search
        # This is needed for LanceDB to vectorize text queries automatically
        """ from lancedb.embeddings import get_registry
        self.embedding_func = get_registry().get("sentence-transformers").create(
            name="all-MiniLM-L6-v2", 
            device=self.device
        ) """
        
        # CLIP model for image search
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def search_by_text(self, query, top_k=5, where=None, min_score=None):
        """
        Hybrid Search: FTS + Vector with optional relevance filtering.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            where: SQL-like filter condition
            min_score: Minimum relevance score. Results with _relevance_score < this will be discarded.
                      For hybrid search, typical values: -5 (strict), -10 (moderate), -15 (loose)
                      Note: Scores are negative in LanceDB hybrid search (less negative = more relevant)
        
        Returns:
            pandas DataFrame with search results
        """
        # Fetch more results initially to account for score filtering
        fetch_limit = top_k * 3 if min_score else top_k
        
        search_builder = self.table.search(query, query_type="hybrid", vector_column_name="text_vector").limit(fetch_limit)
        
        if where:
            search_builder = search_builder.where(where)
        
        results = search_builder.to_pandas()
        
        # Apply relevance score threshold if specified
        if min_score is not None and not results.empty and '_relevance_score' in results.columns:
            original_count = len(results)
            results = results[results['_relevance_score'] >= min_score]
            filtered_count = len(results)
            
            if filtered_count < original_count:
                print(f"[RELEVANCE FILTER] {original_count - filtered_count} results filtered out (score < {min_score})")
        
        # Return only top_k after filtering
        return results.head(top_k)

    def search_by_image(self, image_file, top_k=5):
        image = Image.open(image_file)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = image_features.cpu().numpy()[0].tolist()
        
        results = self.table.search(query_vector, vector_column_name="image_vector") \
            .limit(top_k) \
            .to_pandas()
        return results
    
    def _caption_image_groq(self, image_file):
        """
        Generate a recipe caption from an image using Groq Vision API.
        
        Returns:
            str: Caption describing the dish and ingredients
        """
        try:
            # Initialize Groq client
            client = Groq()
            
            # Encode image to base64
            image = Image.open(image_file)
            # Save to bytes
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Call Groq Vision API
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this food image. Identify the dish name and key ingredients. Be specific and concise. Format: 'Dish Name: [name]. Ingredients: [list]'"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=150,
                temperature=0.3
            )
            
            caption = response.choices[0].message.content.strip()
            print(f"[IMAGE CAPTION] {caption}")
            return caption
            
        except Exception as e:
            print(f"[WARNING] Image captioning failed: {e}")
            return "food dish"  # Fallback
    
    def _reciprocal_rank_fusion(self, results_list, k=60):
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.
        
        Args:
            results_list: List of pandas DataFrames (ranked results)
            k: RRF constant (default 60 is standard)
            
        Returns:
            pandas DataFrame with fused results, sorted by RRF score
        """
        # Dictionary to accumulate RRF scores
        scores = {}
        
        for results_df in results_list:
            for rank, (idx, row) in enumerate(results_df.iterrows(), start=1):
                recipe_id = row['title']  # Use title as unique identifier
                rrf_score = 1.0 / (k + rank)
                
                if recipe_id not in scores:
                    scores[recipe_id] = {'score': 0, 'row': row}
                scores[recipe_id]['score'] += rrf_score
        
        # Convert back to DataFrame
        fused_rows = [
            {**data['row'].to_dict(), '_rrf_score': data['score']}
            for recipe_id, data in sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        ]
        
        return pd.DataFrame(fused_rows)
    
    def search_by_image_hybrid(self, image_file, top_k=5):
        """
        Hybrid image search: Combines CLIP visual similarity with LLM semantic understanding.
        
        Path A: CLIP visual search
        Path B: LLM caption → text search
        Fusion: Reciprocal Rank Fusion (RRF)
        
        Args:
            image_file: Path to uploaded image
            top_k: Number of final results to return
            
        Returns:
            pandas DataFrame with fused results
        """
        print("[HYBRID SEARCH] Starting dual-path search...")
        
        # Path A: CLIP visual search
        print("  [Path A] CLIP visual similarity...")
        visual_results = self.search_by_image(image_file, top_k=10)
        
        # Path B: LLM caption → text search
        print("  [Path B] LLM semantic captioning...")
        caption = self._caption_image_groq(image_file)
        text_results = self.search_by_text(caption, top_k=10)
        
        # Fusion: RRF
        print("  [Fusion] Merging results with RRF...")
        fused_results = self._reciprocal_rank_fusion([visual_results, text_results])
        
        print(f"[HYBRID SEARCH] Complete. Returning top {top_k} results.")
        return fused_results.head(top_k)

    def search_by_ingredients(self, ingredients_list, top_k=5, strict=False):
        """
        Search for recipes based on ingredients.
        
        Args:
            ingredients_list (list or str): List of available ingredients.
            top_k (int): Number of results to return.
            strict (bool): 
                If True (Strict/Subset), returns recipes that require *only* the provided ingredients (Recipe <= User).
                If False (Partial/Any), returns recipes that share *at least one* ingredient, 
                sorted by match count (descending) and then missing count (ascending).
        """
        if isinstance(ingredients_list, str):
            ingredients_list = [i.strip().lower() for i in ingredients_list.split(',')]
        else:
            ingredients_list = [i.strip().lower() for i in ingredients_list]

        user_ingredients = set(ingredients_list)
        
        # 1. Fetch all recipes (optimized: could fetch only needed columns)
        # In a real large DB, we'd use an Inverted Index or FTS for pre-filtering.
        # Since we have ~13k, fetching all ingredients is okay for a demo, 
        # but to be slightly better, we can assume we only want recipes that have *at least one* match if partial.
        # However, for pure python flexibility we iterate.
        # Let's fetch the whole frame for valid logic.
        df = self.table.to_pandas() 

        # 2. Logic
        results = []
        
        for idx, row in df.iterrows():
            # Parse recipe ingredients. They are stored as string representation of list "['a', 'b']"
            try:
                # Safer eval
                recipe_ing_list = ast.literal_eval(row['ingredients'])
                # cleanup recipe ingredients for display/logic
                # recipe_ing_list = ["2 cups milk", "1 egg"]
            except:
                continue
                
            # Fuzzy match logic
            # Check which recipe ingredients are covered by user ingredients
            # We assume user inputs simple terms like "milk", "eggs"
            # Recipe terms are "2 cups milk", "large eggs"
            
            matched_indices = set()
            
            # Check for matches
            # For each recipe ingredient, does it contain ANY user ingredient?
            current_missing = []
            current_matches = 0
            
            for ring in recipe_ing_list:
                ring_lower = ring.lower()
                is_match = False
                for uing in user_ingredients:
                    # check if user ingredient (e.g. 'milk') is in recipe string (e.g. '2 cups milk')
                    # Simple substring match
                    if uing in ring_lower:
                        is_match = True
                        break
                
                if is_match:
                    current_matches += 1
                else:
                    current_missing.append(ring)
            
            match_count = current_matches
            missing_count = len(current_missing)
            
            if strict:
                # Strict: Recipe must be a subset of User (i.e., NO missing ingredients)
                if missing_count == 0 and match_count > 0:
                     results.append({
                        **row, 
                        'match_count': match_count,
                        'missing_count': 0,
                        'missing_ingredients': []
                    })
            else:
                # Partial: Must have at least one match
                if match_count > 0:
                    results.append({
                        **row,
                        'match_count': match_count,
                        'missing_count': missing_count,
                        'missing_ingredients': current_missing
                    })
        
        # 3. Create DataFrame and Sort
        if not results:
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        
        if strict:
            # Sort by match count primarily (maximize usage)
            results_df = results_df.sort_values(by=['match_count'], ascending=[False]).head(top_k)
        else:
            # Sort by match count (desc), then missing count (asc)
            results_df = results_df.sort_values(by=['match_count', 'missing_count'], ascending=[False, True]).head(top_k)
            
        return results_df
