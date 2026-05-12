"""
LLM Enrichment Module for Recipe Database

This module provides functionality to enrich recipe data with:
- visual_description: A short description of the food's appearance
- tags: Keyword tags for cuisine, diet, course, and vibe

Uses Groq's Llama 4 Scout Vision API for image analysis.
"""

import os
import json
import base64
import time
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # Replaces deprecated Llama 3.2 Vision
IMAGES_DIR = "Food Images"

# Rate limit settings (Free tier: 100k tokens/day, ~125 images)
BATCH_SIZE = 5
DELAY_BETWEEN_BATCHES = 2  # seconds

# Combined prompt for efficiency (single API call per image)
ENRICHMENT_PROMPT = """Analyze this food image and the recipe information provided. Return a JSON object with exactly these two fields:

1. "visual_description": A short, vivid sentence describing the food's appearance (colors, textures, presentation). Maximum 20 words.

2. "tags": A list of 5-8 keyword tags covering:
   - Cuisine type (e.g., Italian, Mexican, Asian, American)
   - Diet category (e.g., Vegetarian, Vegan, Gluten-Free, Dairy-Free, Keto)
   - Course type (e.g., Breakfast, Lunch, Dinner, Dessert, Snack, Appetizer)
   - Vibe/occasion (e.g., Comfort Food, Quick Meal, Holiday, Party, Healthy)

Recipe Title: {title}
Ingredients: {ingredients}

Return ONLY valid JSON, no other text. Example format:
{{"visual_description": "Golden crispy potatoes with herbs and sea salt flakes", "tags": ["American", "Side Dish", "Vegetarian", "Comfort Food", "Crispy", "Potatoes"]}}
"""

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Determine the media type based on file extension."""
    ext = image_path.lower().split(".")[-1]
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")

def analyze_recipe_image(client: Groq, image_path: str, title: str, ingredients: str) -> dict:
    default_result = {"visual_description": "", "tags": []}
    
    # Chuẩn bị prompt
    prompt = ENRICHMENT_PROMPT.format(title=title, ingredients=ingredients)
    
    try:
        # Check whether the image file exists or not
        has_image = os.path.exists(image_path)
        
        if has_image:
            # If image: Encode image to base64 and include in the content
            image_base64 = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        else:
            # Text-only fallback
            print(f"  [INFO] No image for '{title}', using text-only enrichment.")
            content = [{"type": "text", "text": prompt}]

        # Gọi API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            response_text = match.group(0)
        # JSON response enhancement
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return {
            "visual_description": str(result.get("visual_description", ""))[:200],
            "tags": result.get("tags", [])[:10]
        }
        
    except Exception as e:
        print(f"  [WARN] Error enriching {title}: {e}")
        return default_result

def batch_enrich_recipes(recipes: list[dict], images_dir: str = IMAGES_DIR) -> list[dict]:
    """
    Enrich a batch of recipes with visual descriptions and tags.
    
    Args:
        recipes: List of recipe dicts with 'id', 'title', 'ingredients', 'image_name'
        images_dir: Directory containing recipe images
        
    Returns:
        List of enriched recipe dicts with 'visual_description' and 'tags' added
    """
    client = Groq()
    enriched_recipes = []
    total = len(recipes)
    
    print(f"\n[LLM] Starting LLM enrichment for {total} recipes...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Delay between batches: {DELAY_BETWEEN_BATCHES}s\n")
    
    for i, recipe in enumerate(recipes):
        image_name = recipe.get("image_name", "")
        if not image_name.lower().endswith(".jpg"):
            image_name += ".jpg"

        image_path = os.path.join(images_dir, image_name)
        
        print(f"[{i+1}/{total}] Processing: {recipe.get('title', 'Unknown')[:50]}...")
        
        enrichment = analyze_recipe_image(
            client,
            image_path,
            recipe.get("title", ""),
            recipe.get("ingredients", "")
        )
        
        # Add enrichment to recipe
        enriched_recipe = recipe.copy()
        enriched_recipe["visual_description"] = enrichment["visual_description"]
        enriched_recipe["tags"] = enrichment["tags"]
        enriched_recipes.append(enriched_recipe)
        
        print(f"   [OK] Tags: {enrichment['tags'][:5]}...")
        
        # Delay between batches to respect rate limits
        if (i + 1) % BATCH_SIZE == 0 and i < total - 1:
            print(f"   [WAIT] Batch complete. Waiting {DELAY_BETWEEN_BATCHES}s...")
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    print(f"\n[DONE] LLM enrichment complete! Processed {len(enriched_recipes)} recipes.")
    return enriched_recipes

if __name__ == "__main__":
    # Test with sample data
    test_recipes = [
        {
            "id": 0,
            "title": "Crispy Salt and Pepper Potatoes",
            "ingredients": "2 pounds new potatoes, 1 tablespoon salt, black pepper, olive oil",
            "image_name": "crispy-salt-and-pepper-potatoes-dan-kluger.jpg"
        },
        {
            "id": 1,
            "title": "Thanksgiving Mac and Cheese",
            "ingredients": "1 cup milk, 2 cups cheese, pasta, butter",
            "image_name": "thanksgiving-mac-and-cheese-erick-williams.jpg"
        }
    ]
    
    enriched = batch_enrich_recipes(test_recipes)
    
    print("\n--- Results ---")
    for r in enriched:
        print(f"\n{r['title']}:")
        print(f"  Description: {r['visual_description']}")
        print(f"  Tags: {r['tags']}")
