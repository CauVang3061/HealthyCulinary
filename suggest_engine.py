"""
Suggestion Engine Module

Get assessment_result from CalorieAssessor and generate suggestions to reduce calories if needed.
- Ingredient Swap: For each dish that exceeds calorie threshold, suggest specific ingredient swaps to cut calories while keeping the dish recognizable.
- Recipe Alternative: For dishes that are significantly over, suggest healthier alternative recipes from the database that have similar tags but at least 20% fewer calories per serving.

Output:
    {
        "triggered_by": ["Beef noodle", "Spring rolls"],
        "swaps": [
            {
                "dish":        "Beef noodle",
                "original_ingredient": "Beef",
                "swap_to":     "Lean beef",
                "reason":      "Less saturated fat, maintains flavor",
                "estimated_saving_kcal": 80,
            },
            ...
        ],
        "alternatives": [
            {
                "original_dish":  "Spring rolls",
                "alternative":    "Vegetable spring rolls",
                "original_cal":   320,
                "alternative_cal": 140,
                "saving_kcal":    180,
                "tags":           ["Vietnamese", "Light"],
                "score":          0.74,
            },
            ...
        ],
        "total_potential_saving": 260,
    }

Usage:
    from calorie_assessor import CalorieAssessor
    from meal_parser import MealParser
    from suggestion_engine import SuggestionEngine

    parser   = MealParser()
    assessor = CalorieAssessor()
    suggester = SuggestionEngine()

    items  = parser.parse_from_text("1 bowl beef noodle, 2 spring rolls")
    result = assessor.assess(items, meal_type="lunch")

    if result["needs_suggestion"]:
        suggestions = suggester.suggest(result)
        print(suggester.format_suggestions(suggestions))
"""

import json
import re
import math
import concurrent.futures
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_SWAP    = "llama-3.3-70b-versatile"   # Text LLM cho ingredient swap
MAX_SWAPS_PER_DISH    = 2    # 2 swap suggestions mỗi món
MAX_ALTERNATIVES      = 3    # 3 recipe alternatives
CALORIE_REDUCTION_PCT = 0.20 # Alternatives must be at least 20% lower in calories
LANCEDB_MIN_SCORE     = 0.015

# Ranking weights cho recipe alternatives
WEIGHT_CALORIE_SAVING = 0.6
WEIGHT_SIMILARITY     = 0.4

SWAP_PROMPT = """You are a nutritionist assistant. A user ate the following dish and it has too many calories.
Suggest ingredient swaps to reduce calories while keeping the dish recognizable and tasty.

Dish: {dish_name}
Estimated calories: {calories} kcal
Portion: {portion}
Known ingredients (if any): {ingredients}

RULES:
1. Return ONLY a valid JSON array of swap objects. No markdown, no explanation.
2. Each swap object must have exactly:
   - "original_ingredient": string — the high-calorie ingredient to replace
   - "swap_to": string — the lower-calorie alternative
   - "reason": string — brief explanation (max 15 words)
   - "estimated_saving_kcal": integer — rough kcal saved per serving
3. Maximum {max_swaps} swaps. Focus on the highest-impact changes.
4. Only suggest realistic, widely available substitutes.
5. If the dish is already healthy or no good swap exists, return empty array [].

Example output:
[
  {{"original_ingredient": "Heavy cream", "swap_to": "Greek yogurt", "reason": "Same creaminess, 60% less fat", "estimated_saving_kcal": 120}},
  {{"original_ingredient": "White rice", "swap_to": "Cauliflower rice", "reason": "Lower carbs, same texture", "estimated_saving_kcal": 100}}
]

Now suggest swaps:"""

class SuggestionEngine:
    
    def __init__(self):
        self._engine = None
        self._groq   = Groq()

    def suggest(self, assessment_result: dict) -> dict:
        if not assessment_result.get("needs_suggestion"):
            return {"triggered_by": [], "swaps": [], "alternatives": [], "total_potential_saving": 0}

        items = assessment_result.get("items", [])
        over_by = assessment_result.get("over_by", 0)

        ranked_items = sorted(
            [i for i in items if i.get("calories_total") is not None],
            key=lambda x: x["calories_total"],
            reverse=True,
        )

        target_items = self._pick_items_to_adjust(ranked_items, over_by)
        triggered_by = [i["name"] for i in target_items]

        print(f"[SUGGESTER] Triggered for: {triggered_by}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_swaps = executor.submit(self._run_swap_branch, target_items)
            future_alts  = executor.submit(self._run_alternative_branch, target_items)

            swaps        = future_swaps.result()
            alternatives = future_alts.result()

        swap_saving = sum(s.get("estimated_saving_kcal", 0) for s in swaps)
        alt_saving  = max((a.get("saving_kcal", 0) for a in alternatives), default=0)
        total_saving = swap_saving + alt_saving

        result = {
            "triggered_by":         triggered_by,
            "swaps":                 swaps,
            "alternatives":          alternatives,
            "total_potential_saving": round(total_saving),
        }

        print(f"[SUGGESTER] Done — {len(swaps)} swaps, {len(alternatives)} alternatives, "
              f"~{total_saving:.0f} kcal potential saving")
        return result

    def _run_swap_branch(self, items: list[dict]) -> list[dict]:
        all_swaps = []
        for item in items:
            swaps = self._get_swaps_for_item(item)
            for s in swaps:
                all_swaps.append({"dish": item["name"], **s})
        return all_swaps

    def _get_swaps_for_item(self, item: dict) -> list[dict]:
        dish_name  = item["name"]
        calories   = item.get("calories_total") or item.get("calories_per_serving") or 0
        portion    = item.get("portion", "1 serving")
        ingredients = item.get("notes", "") or ""

        if calories < 150:
            print(f"  [SWAP SKIP] {dish_name} — only {calories:.0f} kcal, skip")
            return []

        prompt = SWAP_PROMPT.format(
            dish_name=dish_name,
            calories=int(calories),
            portion=portion,
            ingredients=ingredients if ingredients else "unknown",
            max_swaps=MAX_SWAPS_PER_DISH,
        )

        try:
            resp = self._groq.chat.completions.create(
                model=MODEL_SWAP,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            swaps = self._extract_json_list(raw)

            # Validate schema
            valid = []
            for s in swaps:
                if all(k in s for k in ("original_ingredient", "swap_to", "reason", "estimated_saving_kcal")):
                    valid.append({
                        "original_ingredient":   str(s["original_ingredient"]),
                        "swap_to":               str(s["swap_to"]),
                        "reason":                str(s["reason"]),
                        "estimated_saving_kcal": int(s.get("estimated_saving_kcal", 0)),
                    })
            print(f"  [SWAP] {dish_name} → {len(valid)} swaps")
            return valid[:MAX_SWAPS_PER_DISH]

        except Exception as e:
            print(f"  [SWAP][WARN] Failed for '{dish_name}': {e}")
            return []

    def _run_alternative_branch(self, items: list[dict]) -> list[dict]:
        all_alts = []
        seen_titles = set()

        for item in items:
            alts = self._find_alternatives_for_item(item)
            for alt in alts:
                title = alt.get("alternative", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_alts.append(alt)

        # Global rank: calorie_saving × W1 + normalized_score × W2
        all_alts.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_alts[:MAX_ALTERNATIVES]

    def _find_alternatives_for_item(self, item: dict) -> list[dict]:
        dish_name     = item["name"]
        original_cal  = item.get("calories_per_serving") or item.get("calories_total") or 0
        matched_recipe = item.get("matched_recipe")

        if original_cal < 100:
            print(f"  [ALT SKIP] {dish_name} — only {original_cal:.0f} kcal/serving, skip")
            return []

        cal_threshold = original_cal * (1 - CALORIE_REDUCTION_PCT)

        try:
            engine = self._get_engine()

            tags_filter = self._get_tags_filter(matched_recipe, engine)

            cal_filter = f"calories_per_serving < {cal_threshold:.1f} AND calories_per_serving > 0"
            where_clause = f"{cal_filter} AND ({tags_filter})" if tags_filter else cal_filter

            results = engine.search_by_text(
                dish_name,
                top_k=10,
                where=where_clause,
                min_score=LANCEDB_MIN_SCORE,
            )

            if results.empty:
                print(f"  [ALT] No tag-filtered results for '{dish_name}', retrying without tag filter")
                results = engine.search_by_text(
                    dish_name,
                    top_k=10,
                    where=cal_filter,
                    min_score=LANCEDB_MIN_SCORE,
                )

            if results.empty:
                print(f"  [ALT] No alternatives found for '{dish_name}'")
                return []

            if matched_recipe:
                results = results[results["title"].str.lower() != matched_recipe.lower()]

            alternatives = []
            max_relevance = results["_relevance_score"].max() if "_relevance_score" in results.columns else 1.0

            for _, row in results.iterrows():
                alt_cal = row.get("calories_per_serving")
                if alt_cal is None or (isinstance(alt_cal, float) and math.isnan(alt_cal)):
                    continue

                alt_cal = float(alt_cal)
                saving  = original_cal - alt_cal

                rel_score = float(row.get("_relevance_score", 0))
                norm_sim  = rel_score / max_relevance if max_relevance else 0

                norm_saving = min(saving / original_cal, 1.0) if original_cal > 0 else 0

                composite = WEIGHT_CALORIE_SAVING * norm_saving + WEIGHT_SIMILARITY * norm_sim

                tags = row.get("tags", [])
                if not isinstance(tags, list):
                    tags = []

                alternatives.append({
                    "original_dish":   dish_name,
                    "alternative":     row["title"],
                    "original_cal":    round(original_cal, 1),
                    "alternative_cal": round(alt_cal, 1),
                    "saving_kcal":     round(saving, 1),
                    "tags":            tags[:5],
                    "score":           round(composite, 4),
                    "visual_description": str(row.get("visual_description") or ""),
                })

            alternatives.sort(key=lambda x: x["score"], reverse=True)
            print(f"  [ALT] {dish_name} → {len(alternatives)} alternatives found")
            return alternatives[:MAX_ALTERNATIVES]

        except Exception as e:
            print(f"  [ALT][WARN] Failed for '{dish_name}': {e}")
            return []

    def _pick_items_to_adjust(self, ranked_items: list[dict], over_by: float) -> list[dict]:
        selected = []
        accumulated = 0.0
        for item in ranked_items:
            selected.append(item)
            accumulated += item.get("calories_total", 0)
            if accumulated >= over_by:
                break
        return selected

    def _get_engine(self):
        """Lazy-load RecipeSearchEngine."""
        if self._engine is None:
            from search_engine import RecipeSearchEngine
            self._engine = RecipeSearchEngine()
        return self._engine

    def _get_tags_filter(self, matched_recipe: Optional[str], engine) -> Optional[str]:
        if not matched_recipe:
            return None
        try:
            table = engine.table
            df = table.to_pandas()
            rows = df[df["title"].str.lower() == matched_recipe.lower()]
            if rows.empty:
                return None

            tags = rows.iloc[0].get("tags", [])
            if not isinstance(tags, list) or not tags:
                return None

            top_tags = tags[:3]
            tags_str = ", ".join(f"'{t}'" for t in top_tags)
            return f"array_has_any(tags, [{tags_str}])"

        except Exception as e:
            print(f"  [ALT][WARN] Could not get tags for '{matched_recipe}': {e}")
            return None

    def _extract_json_list(self, raw: str) -> list:
        """Robustly extract JSON array từ LLM output."""
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            result = json.loads(raw.strip())
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    def format_suggestions(self, suggestions: dict) -> str:
        """
        Format suggestions into markdown string

        Args:
            suggestions: Output from SuggestionEngine.suggest()

        Returns:
            Markdown string
        """
        if not suggestions["swaps"] and not suggestions["alternatives"]:
            return "There is no need to change! Your meal is already within a healthy calorie range. Great job!"

        lines = []
        total_saving = suggestions.get("total_potential_saving", 0)
        lines.append(f"## Suggestions to reduce ~{total_saving:.0f} kcal")
        lines.append("")

        # Swaps
        if suggestions["swaps"]:
            lines.append("### Swap Ingredients (Keep the Same Dish)")
            lines.append("")
            current_dish = None
            for swap in suggestions["swaps"]:
                dish = swap.get("dish", "")
                if dish != current_dish:
                    lines.append(f"**{dish}**")
                    current_dish = dish
                saving = swap.get("estimated_saving_kcal", 0)
                lines.append(
                    f"- ~~{swap['original_ingredient']}~~ → **{swap['swap_to']}**  "
                    f"_{swap['reason']}_ (−{saving} kcal)"
                )
            lines.append("")

        # --- Nhánh B: Alternatives ---
        if suggestions["alternatives"]:
            lines.append("### Healthier Alternatives")
            lines.append("")
            for alt in suggestions["alternatives"]:
                saving = alt.get("saving_kcal", 0)
                orig_cal = alt.get("original_cal", 0)
                alt_cal  = alt.get("alternative_cal", 0)
                tags_str = ", ".join(alt.get("tags", [])[:3])
                desc = alt.get("visual_description", "")

                lines.append(f"**{alt['alternative']}** _({alt['original_dish']})_")
                if desc:
                    lines.append(f"_{desc}_")
                lines.append(
                    f"🔥 {alt_cal:.0f} kcal vs {orig_cal:.0f} kcal — save **{saving:.0f} kcal**"
                )
                if tags_str:
                    lines.append(f"Tags: {tags_str}")
                lines.append("")

        return "\n".join(lines)

if __name__ == "__main__":
    from meal_parser import MealParser
    from calories_assessor import CalorieAssessor

    parser    = MealParser()
    assessor  = CalorieAssessor()
    suggester = SuggestionEngine()

    print("\n" + "="*60)
    print("TEST 1: Over-energy meal")
    print("="*60)
    items  = parser.parse_from_text(
        "I ate a large bowl of beef pho, 2 spring rolls, and a glass of orange juice for lunch"
    )
    result = assessor.assess(items, meal_type="lunch")

    print(f"\nNeeds suggestion: {result['needs_suggestion']}")
    print(f"Over by: {result['over_by']} kcal\n")

    if result["needs_suggestion"]:
        suggestions = suggester.suggest(result)
        print(suggester.format_suggestions(suggestions))
    else:
        print("There is no need to change! Your meal is already within a healthy calorie range. Great job!")

    print("\n" + "="*60)
    print("TEST 2: Under-energy meal — no suggestion needed")
    print("="*60)
    items2  = parser.parse_from_text("1 bowl of white rice and 1 bowl of vegetable soup")
    result2 = assessor.assess(items2, meal_type="lunch")
    print(f"Needs suggestion: {result2['needs_suggestion']} (expected: False)")