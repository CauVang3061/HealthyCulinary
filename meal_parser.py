"""
Meal Parser Module

Get structured food item lists from free-form meal descriptions.
    [
        {
            "name": "Beef Noodle Soup",
            "portion": "A big bowl",
            "portion_grams": 500,
            "notes": "One spring roll"
        },
        ...
    ]

Usage:
    from meal_parser import MealParser

    parser = MealParser()

    # from text input
    items = parser.parse_from_text("I want to eat 1 big bowl of beef noodle soup and 2 pieces of spring rolls")

    # From image (with caption from Groq Vision)
    items = parser.parse_from_image_caption("Dish: Pho bo. Ingredients: beef, noodles")

    # Validate / normalize
    items = parser.normalize(items)
"""

import json
import re
import base64
import io
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_TEXT = "llama-3.3-70b-versatile"   # text parsing
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"  # image captioning (replaces deprecated Llama 3.2 Vision)

# Normalization map to estimate grams from portion units
PORTION_GRAM_MAP = {
    "big bowl": 500, "small bowl": 250, "bowl": 350,
    "large plate": 600, "small plate": 300, "plate": 450,
    "large cup": 400, "small cup": 150, "cup": 250,
    "piece": 80, "slice": 60,
    "serving": 200, "portion": 200,
    "g": 1, "gram": 1, "ml": 1,
}

TEXT_PARSE_PROMPT = """You are a meal parsing assistant. Your job is to extract a structured list of food items from a user's meal description.

RULES:
1. Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
2. Each item must have exactly these fields:
   - "name": string — the dish or food item name (in the original language if possible)
   - "portion": string — quantity + unit as described (e.g. "One big bowl", "2 pieces", "A large cup")
   - "portion_quantity": number — numeric quantity only (e.g. 1, 2, 0.5)
   - "portion_unit": string — unit only (e.g. "bowl", "piece", "cup")
   - "notes": string — any extra details, "" if none
3. If quantity is not specified, default portion_quantity to 1.
4. If unit is not specified, use "portion" as default unit.
5. Separate combined items (e.g. "beef and noodles" → two items).

EXAMPLES:

Input: "I want to eat 1 big bowl of beef noodle soup and 2 pieces of spring rolls"
Output:
[
  {"name": "Beef Noodle Soup", "portion": "1 big bowl", "portion_quantity": 1, "portion_unit": "big bowl", "notes": ""},
  {"name": "Spring Rolls", "portion": "2 pieces", "portion_quantity": 2, "portion_unit": "piece", "notes": ""}
]

Input: "breakfast with a slice of toast, a cup of coffee, and a small bowl of fruit salad"
Output:
[
  {"name": "Toast", "portion": "1 slice", "portion_quantity": 1, "portion_unit": "slice", "notes": ""},
  {"name": "Coffee", "portion": "1 cup", "portion_quantity": 1, "portion_unit": "cup", "notes": ""},
  {"name": "Fruit Salad", "portion": "1 small bowl", "portion_quantity": 1, "portion_unit": "small bowl", "notes": ""}
]

Now parse this meal description:
"""

IMAGE_PARSE_PROMPT = """You are a meal parsing assistant. You have been given a description of a food image.
Extract a structured list of food items visible in the image.

RULES:
1. Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
2. Each item must have exactly these fields:
   - "name": string — the dish or food item name
   - "portion": string — estimated portion (e.g. "1 serving", "1 bowl")
   - "portion_quantity": number — numeric quantity
   - "portion_unit": string — unit (default "serving" if unclear)
   - "notes": string — any visual details like "large portion", "with sauce", "" if none
3. If quantity is unclear from image, default to 1 serving.
4. List ALL distinct food items visible, including sides and drinks.

Image description:
"""

MANUAL_PARSE_PROMPT = """You are a meal parsing assistant. The user has manually listed food items.
Normalize and structure them into a JSON array.

RULES:
1. Return ONLY a valid JSON array.
2. Each item must have: "name", "portion", "portion_quantity" (number), "portion_unit" (string), "notes" (string).
3. Keep the user's food names as-is.
4. Parse quantity/unit from the user's input if provided.

User input:
"""

class MealParser:
    """
    Parses meal descriptions into structured food item lists.

    Supports three input modes:
      - parse_from_text()         : free-text meal description
      - parse_from_image()        : PIL Image or file-like object → Vision LLM caption → parse
      - parse_from_image_caption(): pre-generated caption string → parse
      - parse_from_manual()       : structured manual input (list of strings)
    """

    def __init__(self):
        self.client = Groq()

    def parse_from_text(self, text: str) -> list[dict]:
        """
        Parse a free-text meal description.

        Args:
            text: e.g. "I want to eat 1 big bowl of beef noodle soup and 2 pieces of spring rolls"

        Returns:
            List of normalized meal item dicts
        """
        print(f"[PARSER] Parsing text: {text[:80]}...")
        raw = self._call_llm_text(TEXT_PARSE_PROMPT + text)
        items = self._extract_json(raw)
        return self.normalize(items)

    def parse_from_image(self, image_file) -> list[dict]:
        """
        Parse from an image file (PIL Image or file-like object).
        Calls Groq Vision to caption the image first, then parses.

        Args:
            image_file: file-like object (e.g. from st.file_uploader)

        Returns:
            List of normalized meal item dicts
        """
        print("[PARSER] Captioning image with Groq Vision...")
        caption = self._caption_image(image_file)
        print(f"[PARSER] Caption: {caption}")
        return self.parse_from_image_caption(caption)

    def parse_from_image_caption(self, caption: str) -> list[dict]:
        """
        Parse from a pre-generated image caption string.
        Useful when caption was already generated by agent.py.

        Args:
            caption: e.g. "Dish: Pho bo. Ingredients: beef, noodles"

        Returns:
            List of normalized meal item dicts
        """
        print(f"[PARSER] Parsing from caption: {caption[:80]}...")
        raw = self._call_llm_text(IMAGE_PARSE_PROMPT + caption)
        items = self._extract_json(raw)
        return self.normalize(items)

    def parse_from_manual(self, entries: list[str]) -> list[dict]:
        """
        Parse from manually entered list of food items.

        Args:
            entries: e.g. ["A big bowl of beef noodle soup", "2 pieces of spring rolls"]

        Returns:
            List of normalized meal item dicts
        """
        combined = "\n".join(f"- {e}" for e in entries)
        print(f"[PARSER] Parsing {len(entries)} manual entries...")
        raw = self._call_llm_text(MANUAL_PARSE_PROMPT + combined)
        items = self._extract_json(raw)
        return self.normalize(items)

    def normalize(self, items: list[dict]) -> list[dict]:
        """
        Normalize and validate parsed items.
        - Fills in missing fields with defaults
        - Estimates portion_grams from portion_unit
        - Sanitizes name strings

        Args:
            items: Raw parsed list from LLM

        Returns:
            Cleaned list of meal item dicts
        """
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            if not name:
                continue

            portion_qty = self._safe_float(item.get("portion_quantity", 1), default=1.0)
            portion_unit = str(item.get("portion_unit", "phần")).strip().lower()
            portion_str = item.get("portion") or f"{portion_qty} {portion_unit}"
            notes = str(item.get("notes", "")).strip()

            # Estimate grams
            gram_per_unit = self._lookup_gram(portion_unit)
            portion_grams = round(portion_qty * gram_per_unit) if gram_per_unit else None

            normalized.append({
                "name": name,
                "portion": str(portion_str).strip(),
                "portion_quantity": portion_qty,
                "portion_unit": portion_unit,
                "portion_grams": portion_grams, # "" if there isn't a known unit to estimate
                "notes": notes,
            })

        print(f"[PARSER] Normalized {len(normalized)} items: {[i['name'] for i in normalized]}")
        return normalized

    def _call_llm_text(self, prompt: str) -> str:
        """Call Groq text LLM and return raw response string."""
        response = self.client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1,   # Low temp: deterministic JSON output
        )
        return response.choices[0].message.content.strip()

    def _caption_image(self, image_file) -> str:
        """
        Send image to Groq Vision and get a food description.
        Returns a caption string describing the dishes in the image.
        """
        try:
            from PIL import Image

            image = Image.open(image_file)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = self.client.chat.completions.create(
                model=MODEL_VISION,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are analyzing a meal photo. "
                                "List ALL distinct food and drink items visible. "
                                "For each item, estimate portion size (e.g. 1 bowl, 2 pieces, 1 glass). "
                                "Format: 'Item 1: [name], approx [quantity] [unit]. Item 2: ...'"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ]
                }],
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[PARSER][WARN] Vision captioning failed: {e}")
            return "Unable to analyze image. Please describe your meal in text."

    def _extract_json(self, raw: str) -> list:
        """
        Robustly extract a JSON array from LLM output.
        Handles markdown fences and extra text before/after the array.
        """
        # Strip markdown fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        # Find the JSON array boundaries
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            result = json.loads(raw.strip())
            if isinstance(result, list):
                return result
            # LLM sometimes wraps in {"items": [...]}
            if isinstance(result, dict):
                for key in ("items", "meals", "foods", "dishes"):
                    if key in result and isinstance(result[key], list):
                        return result[key]
        except json.JSONDecodeError as e:
            print(f"[PARSER][WARN] JSON parse failed: {e}\nRaw: {raw[:200]}")

        return []

    def _safe_float(self, value, default: float = 1.0) -> float:
        """Convert value to float safely, returning default on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _lookup_gram(self, unit: str) -> Optional[float]:
        """
        Look up gram estimate for a portion unit.
        Tries exact match first, then partial match.
        """
        unit_lower = unit.lower().strip()

        # Exact match
        if unit_lower in PORTION_GRAM_MAP:
            return float(PORTION_GRAM_MAP[unit_lower])

        # Partial match (e.g. "bowl large" → "bát lớn")
        for key, val in PORTION_GRAM_MAP.items():
            if key in unit_lower or unit_lower in key:
                return float(val)

        return None   # Unknown unit — calorie_assessor sẽ dùng serving mặc định

if __name__ == "__main__":
    parser = MealParser()

    print("\n" + "="*60)
    print("TEST 1: English text input")
    print("="*60)
    result = parser.parse_from_text(
        "I had a large bowl of mac and cheese, 2 slices of garlic bread, "
        "and a glass of orange juice"
    )
    for item in result:
        print(f"  - {item['name']} | {item['portion']} | ~{item['portion_grams']}g | notes: {item['notes']}")

    print("\n" + "="*60)
    print("TEST 2: Manual entry list")
    print("="*60)
    result = parser.parse_from_manual([
        "A big bowl of beef noodle soup",
        "2 pieces of spring rolls",
        "A small bowl of fruit salad"
    ])
    for item in result:
        print(f"  - {item['name']} | {item['portion']} | ~{item['portion_grams']}g | notes: {item['notes']}")

    print("\n" + "="*60)
    print("TEST 3: Mixed language / edge case")
    print("="*60)
    result = parser.parse_from_text("breakfast: a slice of toast, a cup of coffee, and a small bowl of fruit salad")
    for item in result:
        print(f"  - {item['name']} | {item['portion']} | ~{item['portion_grams']}g")