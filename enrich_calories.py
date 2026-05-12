"""
Resumable calorie enrichment for recipe database.

This script estimates servings from instructions, calls Spoonacular's
analyzeRecipe API, and stores calories per serving plus a Low/Medium/High
label in LanceDB.

Usage:
    python enrich_calories.py
    python enrich_calories.py --limit 50
    python enrich_calories.py --status
"""

import argparse
import ast
import os
import re
import time
from typing import List, Optional

import lancedb
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"
SPOONACULAR_API_URL = "https://api.spoonacular.com/recipes/analyze"
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

# Rate limit handling
BATCH_SIZE = 5
DELAY_BETWEEN_BATCHES = 2  # seconds

# Calorie thresholds (per serving)
LOW_MAX = 400
MEDIUM_MAX = 700


def estimate_servings(instructions: str) -> int:
    """Estimate servings from instructions; fallback to 1."""
    if not instructions:
        return 1

    patterns = [
        r"serves\s+(\d+)(?:\s*[-to]+\s*(\d+))?",
        r"makes\s+(\d+)(?:\s*[-to]+\s*(\d+))?",
        r"yield[s]?\s+(\d+)(?:\s*[-to]+\s*(\d+))?",
        r"servings?\s*[:\-]?\s*(\d+)(?:\s*[-to]+\s*(\d+))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, instructions, re.IGNORECASE)
        if match:
            first = match.group(1)
            if first and first.isdigit():
                return max(1, int(first))

    return 1


def parse_ingredients(ingredients_str: str) -> List[str]:
    """Parse ingredients from CSV string to list of lines."""
    if not ingredients_str:
        return []

    try:
        ingredients = ast.literal_eval(ingredients_str)
        if isinstance(ingredients, list):
            return [str(i).strip() for i in ingredients if str(i).strip()]
    except (ValueError, SyntaxError):
        pass

    # Fallback: split by comma
    return [part.strip() for part in ingredients_str.split(",") if part.strip()]


def categorize_calories(calories: Optional[float]) -> str:
    """Map calories per serving to Low/Medium/High labels."""
    if calories is None:
        return ""
    if calories < LOW_MAX:
        return "Low"
    if calories <= MEDIUM_MAX:
        return "Medium"
    return "High"


def analyze_calories(
    title: str,
    ingredients: List[str],
    instructions: str,
    servings: int,
) -> Optional[float]:
    """Call Spoonacular analyzeRecipe and return calories per serving."""
    if not SPOONACULAR_API_KEY:
        raise RuntimeError("SPOONACULAR_API_KEY is not set")

    payload = {
        "title": title,
        "servings": servings,
        "ingredients": ingredients,
        "instructions": instructions or "",
    }

    response = requests.post(
        SPOONACULAR_API_URL,
        params={
            "apiKey": SPOONACULAR_API_KEY,
            "includeNutrition": "true"
        },
        json=payload,
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Spoonacular error {response.status_code}: {response.text}")

    data = response.json()

    # Primary: nutrition nutrients list
    nutrition = data.get("nutrition", {}) if isinstance(data, dict) else {}
    nutrients = nutrition.get("nutrients", []) if isinstance(nutrition, dict) else []
    for nutrient in nutrients:
        if str(nutrient.get("name", "")).lower() == "calories":
            return float(nutrient.get("amount"))

    # Fallback: top-level calories field
    if "calories" in data:
        return float(data.get("calories"))

    return None


def get_unenriched_recipes(limit: Optional[int] = None) -> List[dict]:
    """Find recipes missing calorie data."""
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()

    if "calories_per_serving" not in df.columns:
        raise RuntimeError(
            "Database schema missing calories_per_serving. Re-run ingest.py to rebuild the table."
        )

    unenriched = df[df["calories_per_serving"].isna()]
    if limit:
        unenriched = unenriched.head(limit)

    recipes = []
    for _, row in unenriched.iterrows():
        recipes.append(
            {
                "id": int(row["id"]),
                "title": row["title"],
                "ingredients": row["ingredients"],
                "instructions": row["instructions"],
            }
        )
    return recipes


def update_calories(enriched: List[dict]) -> None:
    """Persist calorie data to LanceDB."""
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)

    for i, item in enumerate(enriched, start=1):
        tbl.update(
            where=f"id = {item['id']}",
            values={
                "servings": item.get("servings", 1),
                "calories_per_serving": item.get("calories_per_serving"),
                "calorie_level": item.get("calorie_level", ""),
            },
        )

        if i % 10 == 0:
            print(f"  Updated {i}/{len(enriched)} recipes...")


def get_progress() -> dict:
    """Get calorie enrichment progress."""
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()

    total = len(df)
    if "calories_per_serving" not in df.columns:
        return {
            "total": total,
            "enriched": 0,
            "remaining": total,
            "percent_complete": 0,
            "schema_outdated": True,
        }

    enriched = len(df[df["calories_per_serving"].notna()])
    remaining = total - enriched
    return {
        "total": total,
        "enriched": enriched,
        "remaining": remaining,
        "percent_complete": round(enriched / total * 100, 1) if total > 0 else 0,
        "schema_outdated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable calorie enrichment")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    progress = get_progress()
    print("\n" + "=" * 50)
    print("Calorie Enrichment Progress")
    print("=" * 50)
    print(f"  Total recipes:    {progress['total']}")
    print(f"  Enriched:         {progress['enriched']}")
    print(f"  Remaining:        {progress['remaining']}")
    print(f"  Progress:         {progress['percent_complete']}%")
    print("=" * 50)

    if progress.get("schema_outdated"):
        print("\n[WARN] Schema missing calorie fields. Re-run ingest.py to rebuild the table.")
        return

    if args.status:
        return

    if progress["remaining"] == 0:
        print("\n[DONE] All recipes already have calorie data.")
        return

    recipes = get_unenriched_recipes(limit=args.limit)
    if not recipes:
        print("\n[DONE] No recipes to enrich.")
        return

    enriched = []
    for i, recipe in enumerate(recipes, start=1):
        title = recipe["title"]
        ingredients = parse_ingredients(recipe["ingredients"])
        instructions = recipe["instructions"]
        servings = estimate_servings(instructions)

        print(f"[{i}/{len(recipes)}] Analyzing: {title[:60]}...")
        try:
            calories = analyze_calories(title, ingredients, instructions, servings)
            enriched.append(
                {
                    "id": recipe["id"],
                    "servings": servings,
                    "calories_per_serving": calories,
                    "calorie_level": categorize_calories(calories),
                }
            )
            print(f"   [OK] Calories/serving: {calories}")
        except Exception as exc:
            print(f"   [WARN] Failed: {exc}")
            enriched.append(
                {
                    "id": recipe["id"],
                    "servings": servings,
                    "calories_per_serving": None,
                    "calorie_level": "",
                }
            )

        if i % BATCH_SIZE == 0 and i < len(recipes):
            print(f"   [WAIT] Batch complete. Waiting {DELAY_BETWEEN_BATCHES}s...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    print("\n[DB] Updating records with calorie data...")
    update_calories(enriched)
    print("[DONE] Calorie enrichment complete.")


if __name__ == "__main__":
    main()