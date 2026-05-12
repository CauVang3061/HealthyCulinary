"""
Calorie Assessor Module

Get food list from MealParser, search for calories, aggregate and assess to the WHO meal targets.

Lookup strategy:
  1. LanceDB  — find most relevant recipe, get enriched calories_per_serving
  2. Spoonacular API — fallback if there is no in DB match or if calories not enriched yet
  3. Groq LLM estimate — fallback if Spoonacular fails or is not configured

Output:
    {
        "meal_type": "lunch",
        "items": [
            {
                "name": "Beef noodle soup",
                "portion": "1 big bowl",
                "calories_per_serving": 420.0,
                "calories_total": 420.0,
                "source": "lancedb",
                "matched_recipe": "Beef Noodle Soup",
                "confidence": "high"
            },
            ...
        ],
        "total_calories": 860.0,
        "who_target": 700,
        "who_status": "over",
        "over_by": 160.0,
        "percent_daily": 43.0,
        "calorie_level": "Medium",
        "needs_suggestion": True
    }

Usage:
    from meal_parser import MealParser
    from calorie_assessor import CalorieAssessor

    parser = MealParser()
    assessor = CalorieAssessor()

    items = parser.parse_from_text("1 big bowl of beef noodle soup, 2 spring rolls, 1 small bowl of fruit salad")
    result = assessor.assess(items, meal_type="lunch")
    print(result)
"""

import os
import re
import math
import requests
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_PATH       = "data/lancedb"
TABLE_NAME    = "recipes"

SPOONACULAR_API_URL = "https://api.spoonacular.com/recipes/analyze"
SPOONACULAR_KEY     = os.getenv("SPOONACULAR_API_KEY")

# WHO/FDA daily target: 2000 kcal/day
DAILY_TARGET = 2000

# Meal-type calorie targets (kcal) — WHO-based distribution
MEAL_TARGETS = {
    "breakfast": 500,   # ~25%
    "lunch":     700,   # ~35%
    "dinner":    700,   # ~35%
    "snack":     200,   # ~10%
    "general":   600,   # fallback
}

# Suggestion threshold: trigger if target >10%
SUGGEST_THRESHOLD_PCT = 0.10

# LanceDB min relevance score
LANCEDB_MIN_SCORE = 0.020

# Calorie level thresholds
LEVEL_LOW_MAX    = 400
LEVEL_MEDIUM_MAX = 800

class CalorieAssessor:
    """
    Implementation of calorie assessment logic.
    """

    def __init__(self):
        self._engine  = None
        self._groq    = Groq()

    def assess(self, items: list[dict], meal_type: str = "general") -> dict:
        """
        Calories assessment for a meal based on its items and meal type.

        Args:
            items:     Output from MealParser.normalize()
            meal_type: "breakfast" | "lunch" | "dinner" | "snack" | "general"

        Returns:
            Assessment dict — xem module docstring để biết format
        """
        meal_type = meal_type.lower().strip()
        if meal_type not in MEAL_TARGETS:
            meal_type = "general"

        print(f"[ASSESSOR] Assessing {len(items)} items for meal_type='{meal_type}'")

        # Lookup calo
        assessed_items = [self._lookup_item(item) for item in items]

        # Aggregation
        total = sum(i["calories_total"] for i in assessed_items if i["calories_total"] is not None)
        target = MEAL_TARGETS[meal_type]
        over_by = round(total - target, 1)
        percent_daily = round((total / DAILY_TARGET) * 100, 1)
        who_status = (
            "over"  if over_by >  target * SUGGEST_THRESHOLD_PCT else
            "under" if over_by < -target * SUGGEST_THRESHOLD_PCT else
            "ok"
        )
        needs_suggestion = who_status == "over"
        calorie_level = _categorize(total)

        result = {
            "meal_type":        meal_type,
            "items":            assessed_items,
            "total_calories":   round(total, 1),
            "who_target":       target,
            "who_status":       who_status,
            "over_by":          over_by,
            "percent_daily":    percent_daily,
            "calorie_level":    calorie_level,
            "needs_suggestion": needs_suggestion,
        }

        self._log_summary(result)
        return result

    def lookup_single(self, food_name: str, portion_quantity: float = 1.0) -> dict:
        """
        Lookup calories for a single food item (no need to assess the entire meal).
        Convenient for debugging or quick checks from the UI.

        Returns:
            {"name", "calories_per_serving", "calories_total", "source", "confidence"}
        """
        item = {
            "name": food_name,
            "portion": f"{portion_quantity} serving",
            "portion_quantity": portion_quantity,
            "portion_unit": "serving",
            "portion_grams": None,
            "notes": "",
        }
        return self._lookup_item(item)

    def _lookup_item(self, item: dict) -> dict:
        """
        Lookup calories for a single item:
          1. LanceDB recipe match
          2. Spoonacular API
          3. Groq LLM estimate
        """
        name  = item["name"]
        qty   = float(item.get("portion_quantity") or 1.0)
        grams = item.get("portion_grams")

        result_base = {
            **item,
            "calories_per_serving": None,
            "calories_total":       None,
            "source":               None,
            "matched_recipe":       None,
            "confidence":           "low",
        }

        # Strategy 1: LanceDB
        db_hit = self._lookup_lancedb(name)
        if db_hit:
            cal_per_serving = db_hit["calories_per_serving"]
            cal_total = self._scale_calories(cal_per_serving, qty, grams, db_hit.get("servings", 1))
            print(f"  [DB HIT] {name} → {cal_per_serving:.0f} kcal/serving (recipe: {db_hit['title']})")
            return {
                **result_base,
                "calories_per_serving": round(cal_per_serving, 1),
                "calories_total":       round(cal_total, 1),
                "source":               "lancedb",
                "matched_recipe":       db_hit["title"],
                "confidence":           "high",
            }

        # Strategy 2: Spoonacular
        if SPOONACULAR_KEY:
            spoon_cal = self._lookup_spoonacular(name, grams)
            if spoon_cal is not None:
                cal_total = spoon_cal * qty
                print(f"  [SPOONACULAR] {name} → {spoon_cal:.0f} kcal/serving")
                return {
                    **result_base,
                    "calories_per_serving": round(spoon_cal, 1),
                    "calories_total":       round(cal_total, 1),
                    "source":               "spoonacular",
                    "matched_recipe":       None,
                    "confidence":           "medium",
                }
        else:
            print(f"  [SKIP] Spoonacular key not set, skipping for '{name}'")

        # Strategy 3: Groq LLM estimate
        llm_cal = self._lookup_llm_estimate(name, item.get("portion", "1 serving"))
        if llm_cal is not None:
            cal_total = llm_cal * qty
            print(f"  [LLM ESTIMATE] {name} → {llm_cal:.0f} kcal/serving (estimated)")
            return {
                **result_base,
                "calories_per_serving": round(llm_cal, 1),
                "calories_total":       round(cal_total, 1),
                "source":               "llm_estimate",
                "matched_recipe":       None,
                "confidence":           "low",
            }

        # Hoàn toàn không tra được
        print(f"  [MISS] Could not estimate calories for '{name}'")
        return result_base

    def _get_engine(self):
        if self._engine is None:
            from search_engine import RecipeSearchEngine
            self._engine = RecipeSearchEngine()
        return self._engine

    def _lookup_lancedb(self, food_name: str) -> Optional[dict]:
        try:
            engine  = self._get_engine()
            results = engine.search_by_text(
                food_name,
                top_k=1,
                min_score=LANCEDB_MIN_SCORE,
            )

            if results.empty:
                return None

            row = results.iloc[0]
            cal = row.get("calories_per_serving")

            if cal is None or (isinstance(cal, float) and math.isnan(cal)):
                print(f"  [DB] Found '{row['title']}' but calories not enriched yet")
                return None

            return {
                "title":              row["title"],
                "calories_per_serving": float(cal),
                "servings":           int(row.get("servings") or 1),
            }

        except Exception as e:
            print(f"  [DB][WARN] LanceDB lookup failed for '{food_name}': {e}")
            return None

    def _lookup_spoonacular(self, food_name: str, grams: Optional[float]) -> Optional[float]:
        try:
            servings = 1
            ingredients = [f"{int(grams)}g {food_name}"] if grams else [food_name]

            payload = {
                "title": food_name,
                "servings": servings,
                "ingredients": ingredients,
                "instructions": "",
            }
            resp = requests.post(
                SPOONACULAR_API_URL,
                params={
                    "apiKey": SPOONACULAR_KEY,
                    "includeNutrition": True,
                    "language": "en",
                },
                json=payload,
                timeout=15,
            )

            if resp.status_code >= 400:
                print(f"  [SPOONACULAR][WARN] {resp.status_code}: {resp.text[:100]}")
                return None

            data = resp.json()
            nutrition = data.get("nutrition", {})
            for nutrient in nutrition.get("nutrients", []):
                if str(nutrient.get("name", "")).lower() == "calories":
                    return float(nutrient["amount"]) / servings

            if "calories" in data:
                return float(data["calories"]) / servings

        except Exception as e:
            print(f"  [SPOONACULAR][WARN] Failed for '{food_name}': {e}")

        return None

    def _lookup_llm_estimate(self, food_name: str, portion: str) -> Optional[float]:
        prompt = (
            f"Estimate the calories for this food item:\n"
            f"  Food: {food_name}\n"
            f"  Portion: {portion}\n\n"
            f"Reply with ONLY a single integer number representing total calories for that portion. "
            f"No units, no explanation, just the number. "
            f"If you cannot estimate, reply with 0."
        )
        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r'\d+', text)
            if match:
                val = float(match.group(0))
                return val if val > 0 else None
        except Exception as e:
            print(f"  [LLM ESTIMATE][WARN] Failed for '{food_name}': {e}")

        return None

    def _scale_calories(
        self,
        cal_per_serving: float,
        qty: float,
        grams: Optional[float],
        recipe_servings: int,
    ) -> float:
        if grams and recipe_servings:
            # 350g is a heuristic average serving size for mixed dishes
            gram_per_serving = 350.0
            scale = grams / (recipe_servings * gram_per_serving)
            return cal_per_serving * recipe_servings * scale
        return cal_per_serving * qty

    def _log_summary(self, result: dict) -> None:
        status_icon = {"ok": "✓", "over": "!", "under": "↓"}.get(result["who_status"], "?")
        print(
            f"[ASSESSOR] {status_icon} Total: {result['total_calories']} kcal "
            f"| Target: {result['who_target']} kcal ({result['meal_type']}) "
            f"| {result['who_status'].upper()} by {abs(result['over_by'])} kcal "
            f"| {result['percent_daily']}% daily "
            f"| Suggest: {result['needs_suggestion']}"
        )

def _categorize(total_calories: float) -> str:
    if total_calories <= LEVEL_LOW_MAX:
        return "Low"
    if total_calories <= LEVEL_MEDIUM_MAX:
        return "Medium"
    return "High"


def format_assessment_report(result: dict) -> str:
    lines = []

    # Header
    meal_label = result["meal_type"].capitalize()
    status_map = {
        "ok":    ("✅ Good"),
        "over":  ("⚠️ Over-energy", f"Need to lower the calories down"),
        "under": ("ℹ️ Under-energy", f"Need to increase the calories up"),
    }
    status_title, status_desc = status_map.get(result["who_status"], ("?", ""))

    lines.append(f"## Calories summary — {meal_label}")
    lines.append(f"**{status_title}** — {status_desc}")
    lines.append("")

    # Item breakdown
    lines.append("### Table details")
    lines.append("| Dish | Portion | Calories | Source |")
    lines.append("|------|---------|----------|--------|")

    source_label = {"lancedb": "DB", "spoonacular": "Spoonacular", "llm_estimate": "Estimate"}
    for item in result["items"]:
        cal = item["calories_total"]
        cal_str = f"{cal:.0f} kcal" if cal is not None else "N/A"
        src = source_label.get(item.get("source", ""), "?")
        conf = item.get("confidence", "")
        conf_badge = " ⚠️" if conf == "low" else ""
        lines.append(f"| {item['name']} | {item['portion']} | {cal_str}{conf_badge} | {src} |")

    lines.append("")

    # Summary bar
    pct = result["percent_daily"]
    filled = int(pct / 5)    # 1 block = 5%
    bar = "█" * min(filled, 20) + "░" * max(0, 20 - filled)
    lines.append(f"**Total:** {result['total_calories']:.0f} kcal "
                 f"/ target {result['who_target']} kcal ({meal_label})")
    lines.append(f"**% daily:** [{bar}] {pct:.1f}% / 2000 kcal")
    lines.append(f"**Level:** {result['calorie_level']}")

    # Confidence note nếu có ước tính
    low_conf = [i["name"] for i in result["items"] if i.get("confidence") == "low"]
    if low_conf:
        lines.append("")
        lines.append(f"_⚠️ Calories of {', '.join(low_conf)} are estimated — may not be accurate._")

    return "\n".join(lines)

if __name__ == "__main__":
    from meal_parser import MealParser

    parser   = MealParser()
    assessor = CalorieAssessor()

    print("\n" + "="*60)
    print("TEST 1: English lunch")
    print("="*60)
    items = parser.parse_from_text(
        "I ate a large bowl of beef pho, 2 spring rolls, and a glass of orange juice for lunch"
    )
    result = assessor.assess(items, meal_type="lunch")
    print(format_assessment_report(result))

    print("\n" + "="*60)
    print("TEST 2: English breakfast")
    print("="*60)
    items = parser.parse_from_text("I had a slice of toast with butter and a cup of coffee for breakfast")
    result = assessor.assess(items, meal_type="breakfast")
    print(format_assessment_report(result))

    print("\n" + "="*60)
    print("TEST 3: Single item lookup")
    print("="*60)
    hit = assessor.lookup_single("Mac and cheese", portion_quantity=1.5)
    print(f"  {hit['name']}: {hit['calories_total']} kcal total | source: {hit['source']} | confidence: {hit['confidence']}")