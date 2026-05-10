import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import shutil
from typing import List, Optional

# --- Configuration ---
DATA_PATH = "Food Ingredients and Recipe Dataset with Image Name Mapping Final.csv"
IMAGES_DIR = "Food Images"
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

# --- Models ---
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Registry for Text Embedding (MiniLM)
# This allows LanceDB to handle vectorization automatically for text search
embedding_func = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2", device=device)

# --- Schema Definition ---
class Recipe(LanceModel):
    id: int
    title: str
    ingredients: str
    instructions: str
    image_name: str
    # LLM Enrichment fields (added in Phase 2)
    visual_description: str = ""  # Short description of food appearance
    tags: List[str] = []  # Tags: cuisine, diet, course, vibe
    # Calorie enrichment fields (optional)
    servings: int = 1
    calories_per_serving: Optional[float] = None
    calorie_level: str = ""
    # The source field for text search (Combined Title + Ingredients + Instructions)
    search_text: str = embedding_func.SourceField()
    # The auto-generated vector field
    text_vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()
    # Manually computed image vector (CLIP)
    image_vector: Vector(512)

def get_image_embedding(image_name):
    if not image_name.lower().endswith(".jpg"):
        image_name = f"{image_name}.jpg"

    image_path = os.path.join(IMAGES_DIR, image_name)
    if not os.path.exists(image_path):
        return [0.0] * 512

    
    try:
        image = Image.open(image_path)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
        return [0.0] * 512

def process_data():
    if os.path.exists(DB_PATH):
        # We need to recreate the table with the new schema
        shutil.rmtree(DB_PATH) 

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    
    # Cleaning
    initial_len = len(df)
    df = df.dropna(subset=['Title', 'Instructions'])
    print(f"Dropped {initial_len - len(df)} rows with missing Title or Instructions.")
    
    df = df.reset_index(drop=True)
    
    # Connect to LanceDB
    db = lancedb.connect(DB_PATH)
    
    data = []
    print(f"Processing {len(df)} recipes...")
    
    for index, row in df.iterrows():
        image_name = row['Image_Name']
        if pd.isna(image_name):
             image_name = "placeholder.jpg"
        
        # Combine text for the registry embedding
        combined_text = f"{row['Title']} {row['Ingredients']} {row['Instructions']}"
        
        # Image embedding is still manual
        image_embedding = get_image_embedding(image_name)
        
        data.append({
            "id": index,
            "title": row['Title'],
            "ingredients": row['Ingredients'],
            "instructions": row['Instructions'],
            "image_name": str(image_name),
            "visual_description": "",  # Populated in Phase 2
            "tags": [],  # Populated in Phase 2
            "servings": 1,
            "calories_per_serving": None,
            "calorie_level": "",
            "search_text": combined_text, # This will be vectorized automatically into 'text_vector'
            "image_vector": image_embedding
        })
        
        if index > 0 and index % 100 == 0:
            print(f"Processed {index} records...")

    print("Creating LanceDB table with Hybrid Search support...")
    # Create table using the Pydantic schema
    tbl = db.create_table(TABLE_NAME, schema=Recipe, mode="overwrite")
    
    # Add data
    tbl.add(data)
    
    print("Creating FTS index on 'search_text'...")
    # Create Full Text Search index for Hybrid Search
    tbl.create_fts_index("search_text")
    
    print("\n" + "="*50)
    print("Phase 1 (Local Pass) complete!")
    print("="*50)
    
    # --- Phase 2: LLM Enrichment ---
    # Phase 2 is now a SEPARATE script for resumable enrichment.
    # This allows you to run it multiple times to handle rate limits.
    
    print("\n" + "-"*50)
    print("NEXT STEP: Run LLM Enrichment (Phase 2)")
    print("-"*50)
    print("To enrich recipes with visual descriptions and tags, run:")
    print("")
    print("  python enrich_recipes.py           # Enrich all remaining")
    print("  python enrich_recipes.py --limit 50  # Enrich 50 at a time")
    print("  python enrich_recipes.py --status    # Check progress")
    print("")
    print("This script is RESUMABLE - run it multiple times to complete")
    print("the full dataset while respecting API rate limits.")
    print("-"*50)
    
    print("\n" + "="*50)
    print("Ingestion complete!")
    print("="*50)

if __name__ == "__main__":
    process_data()
