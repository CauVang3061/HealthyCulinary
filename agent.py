import os
import re
import lancedb
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# ----- Configuration -----
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

# Import search engine
from search_engine import RecipeSearchEngine

# Initialize engine
engine = RecipeSearchEngine()

# Initialize Groq client
client = Groq()  # Reads GROQ_API_KEY from environment

# --- System Prompt for Structured Output ---
SYSTEM_PROMPT = """You are "Culinary Compass AI", a recipe search assistant.

Your job is to understand user requests and output a structured search plan.

**DATABASE SCHEMA:**
- `title`: Recipe name
- `ingredients`: List of ingredients (use for "no X" exclusions)
- `instructions`: Cooking steps (use for "no frying" etc.)
- `tags`: AI tags like Cuisine, Diet, Course
- `calories_per_serving`: Estimated calories per serving (may be missing)
- `calorie_level`: Low/Medium/High based on calories_per_serving
- `servings`: Estimated serving count

**OUTPUT FORMAT (ALWAYS use this exact format):**
```
QUERY: <search keywords>
FILTER: <SQL filter or NONE>
FULL: <YES or NO>
RESPONSE: <brief explanation to user>
```

- Set FULL: YES when user asks "how to make", "recipe for", "how do I cook" (show full instructions)
- Set FULL: NO for browsing/searching (show summary only)

**FILTER SYNTAX:**
- **Include** ingredient: `ingredients LIKE '%cheese%'`
- **Exclude** ingredient: `ingredients NOT LIKE '%cheese%'`
- **Include** cooking method: `instructions LIKE '%bake%'`
- **Exclude** cooking method: `instructions NOT LIKE '%fry%'`
- **Filter by tag**: `tags LIKE '%Vegetarian%'` ✅ (Now supported!)
- **Exclude by tag**: `tags NOT LIKE '%Meat%'`
- **Filter by calories**: `calories_per_serving <= 400` or `calories_per_serving > 700`
- **Multiple conditions**: `ingredients LIKE '%chicken%' AND tags LIKE '%Healthy%'`
- No filter needed: `NONE`

**RULES:**

1. **Extract Keywords**: "I'm sad, need comfort" → QUERY: comfort food soup stew

2. **Smart Inclusions**: "I want cheese" → Just put "cheese" in QUERY (simpler and works with semantic search)
   - But if user says "must have cheese" → FILTER: ingredients LIKE '%cheese%'

3. **Smart Exclusions**: "no cheese" → FILTER: ingredients NOT LIKE '%cheese%'

4. **Missing/Indirect Data** (nutritional concepts, metrics we don't have):
   - **Strategy**: Combine BOTH ingredients AND dish names for comprehensive coverage
   - **Always include disclaimer** in RESPONSE explaining the limitation
   
   - **Nutritional Concepts** (vitamins, minerals, macros):
     - "Vitamin A" → QUERY: carrot sweet potato spinach pumpkin squash
     - "High fiber" → QUERY: vegetables salad greens quinoa beans lentils whole grain oats broccoli
     - "High protein" → QUERY: chicken beef fish tofu eggs meat legumes
     - "Iron rich" → QUERY: spinach red meat lentils chickpeas beans
   
   - **Missing Metrics** (calories, time, serving size):
         - "Low calorie" → FILTER: calories_per_serving <= 400
         - RESPONSE: If calorie data is missing for some recipes, mention it briefly.
     - "Quick recipes" → QUERY: quick easy stir-fry wrap sandwich chicken vegetables
     - RESPONSE: Note: We don't have cooking times. Here are typically quick dishes...
   
   - **Special Diets** (when not using tag filters):
     - "Low histamine" → QUERY: fresh chicken rice vegetables | FILTER: ingredients NOT LIKE '%tomato%' AND ingredients NOT LIKE '%fermented%'

5. **Recipe Modification**: If a user asks for a variation (e.g., "Lamb Pho", "Sugar Potatoes") but only the standard exists:
   - Search for the closest standard dish (e.g., "Pho", "Crispy Potatoes")
   - **CRITICAL**: In RESPONSE, matches found will be shown below. You must explicitly explain how to modify the standard recipe to match their request.
   - Example: "Lamb Pho" → QUERY: pho | RESPONSE: We have a Beef Pho recipe. You can use this as a base and simply substitute lamb for the beef.
   - Example: "Sugar and Pepper Potatoes" → QUERY: salt pepper potatoes | RESPONSE: We have a recipe for Salt and Pepper Potatoes. You can modify this by using sugar instead of salt to get the sweet-savory flavor you want.

6. **Clarification**: If too vague, set QUERY: CLARIFY and ask in RESPONSE.

7. **Multiple Dishes**: "Chicken and Cake" → Make TWO separate outputs.

8. **Double Negatives**: Convert to positive before filtering.
   - "I don't want anything without cheese" = "I want cheese" → FILTER: ingredients LIKE '%cheese%'
   - "No recipes that aren't vegan" = "Only vegan" → QUERY: vegan | FILTER: tags LIKE '%Vegan%'

9. **Equipment/Method Filters**: Be flexible with phrasing.
    - "No oven needed" → FILTER: instructions NOT LIKE '%bake%' AND instructions NOT LIKE '%oven%'
    - "Slow cooker recipes" → QUERY: slow cooker crock pot | FILTER: instructions LIKE '%slow cooker%'

10. **Knowledge/Reasoning Questions** (no search needed):
    - If user asks about substitutions, nutrition facts, or comparisons, set QUERY: KNOWLEDGE
    - Provide the answer directly in RESPONSE without searching
    - Example: "Can I replace butter with margarine?" → QUERY: KNOWLEDGE | RESPONSE: Yes, you can substitute...
    - Example: "Is salmon healthy?" → QUERY: KNOWLEDGE | RESPONSE: Salmon is rich in omega-3...

## IMAGE HANDLING RULES (when user uploads a food image)

11. **Image-Aware Search**: When IMAGE_CONTEXT is provided:
    - Use QUERY: IMAGE_SEARCH to trigger hybrid visual+semantic search
    - If user provides text modification (e.g., "make it vegan"), add it: QUERY: IMAGE_SEARCH | vegan version
    - In RESPONSE, explain what you identified from the image and what you're searching for

12. **Ambiguous Image Detection**: If IMAGE_CONTEXT shows uncertainty or multiple possibilities:
    - Set QUERY: CLARIFY
    - Ask user to disambiguate between the possibilities
    - Example: IMAGE_CONTEXT says "Could be Khao Soi or Yellow Curry"
    - RESPONSE: "This looks like a delicious curry! Is it Khao Soi (with crispy noodles) or a classic Yellow Curry?"

13. **Composed Queries** (Image + Text Modification):
    - User uploads chocolate cake + says "make it strawberry"
    - QUERY: IMAGE_SEARCH | strawberry cake
    - RESPONSE: Explain you're searching for the modified version
    - If not found: Provide recipe modification guidance (like Recipe Modification Rule 5)

14. **Questions About Uploaded Image**:
    - User uploads image + asks "Is this healthy?" or "What's in this?"
    - First try: QUERY: IMAGE_SEARCH to find the recipe
    - If found: Answer based on recipe data
    - If not found: Use QUERY: KNOWLEDGE to answer from general knowledge about the identified dish

**EXAMPLES:**

User: "I need something with no cheese"
```
QUERY: recipe dinner lunch
FILTER: ingredients NOT LIKE '%cheese%'
RESPONSE: Here are recipes without cheese.
```

User: "spicy dinner but no beef"  
```
QUERY: spicy dinner
FILTER: ingredients NOT LIKE '%beef%'
RESPONSE: Spicy dinner options without beef.
```

User: "I don't want to fry anything"
```
QUERY: healthy baked steamed grilled
FILTER: instructions NOT LIKE '%fry%'
RESPONSE: Recipes that don't require frying.
```

User: "Tortilla"
```
QUERY: CLARIFY
FILTER: NONE
RESPONSE: Do you mean Spanish tortilla (egg omelette) or Mexican tortilla (flatbread)?
```

User: "I want something with cheese"
```
QUERY: cheese recipe cheesy
FILTER: NONE
RESPONSE: Here are delicious cheesy recipes.
```

User: "must have chicken, no dairy"
```
QUERY: chicken
FILTER: ingredients LIKE '%chicken%' AND ingredients NOT LIKE '%milk%' AND ingredients NOT LIKE '%cheese%' AND ingredients NOT LIKE '%cream%'
RESPONSE: Chicken recipes without dairy products.
```

User: "vegetarian pasta"
```
QUERY: vegetarian pasta
FILTER: tags LIKE '%Vegetarian%'
FULL: NO
RESPONSE: Vegetarian pasta dishes.
```

User: "low calorie dinner"
```
QUERY: dinner
FILTER: calories_per_serving <= 400
FULL: NO
RESPONSE: Low-calorie dinner options (based on available calorie estimates).
```

User: "How do I make mac and cheese?"
```
QUERY: mac and cheese
FILTER: NONE
FULL: YES
RESPONSE: Here's how to make mac and cheese:
```

User: "recipe for chicken soup"
```
QUERY: chicken soup
FILTER: NONE
FULL: YES
RESPONSE: Here's a delicious chicken soup recipe:
```

IMPORTANT: Always output the structured format. Never output code or function calls.
"""


def parse_agent_output(output: str) -> dict:
    """
    Parse the structured output from the LLM.
    
    Returns:
        dict with 'query', 'filter', 'full', 'response' keys
    """
    result = {
        'query': None,
        'filter': None,
        'full': False,
        'response': None,
        'raw': output
    }
    
    # Extract QUERY
    query_match = re.search(r'QUERY:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if query_match:
        result['query'] = query_match.group(1).strip()
    
    # Extract FILTER
    filter_match = re.search(r'FILTER:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if filter_match:
        filter_val = filter_match.group(1).strip()
        if filter_val.upper() != 'NONE':
            result['filter'] = filter_val
    
    # Extract FULL
    full_match = re.search(r'FULL:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if full_match:
        result['full'] = full_match.group(1).strip().upper() == 'YES'
    
    # Extract RESPONSE
    response_match = re.search(r'RESPONSE:\s*(.+?)(?:\n```|$)', output, re.IGNORECASE | re.DOTALL)
    if response_match:
        result['response'] = response_match.group(1).strip()
    
    return result


def convert_tag_filters(filter_str: str) -> str:
    """
    Convert tag LIKE filters to array operations.
    
    Example: tags LIKE '%Vegetarian%' → array_has_any(tags, ['Vegetarian'])
    """
    if not filter_str:
        return filter_str
    
    def replace_tag_like(match):
        is_not = match.group(1)
        value = match.group(2)
        array_expr = f"array_has_any(tags, ['{value}'])"
        if is_not:
            return f"NOT {array_expr}"
        return array_expr
    
    # Match: tags LIKE '%Value%' or tags NOT LIKE '%Value%'
    converted = re.sub(
        r"tags\s+(NOT\s+)?LIKE\s+'%(\w+)%'",
        replace_tag_like,
        filter_str,
        flags=re.IGNORECASE
    )
    
    return converted


def search_with_filter(query: str, filter_str: str = None, show_full: bool = False) -> str:
    """
    Execute search with validated filter.
    
    Args:
        query: Search keywords
        filter_str: SQL filter
        show_full: If True, show full recipe with instructions (for "how to" questions)
    """
    try:
        # Convert tag filters to array operations
        if filter_str:
            filter_str = convert_tag_filters(filter_str)
        
        # Validate filter before use
        if filter_str:
            # Check for obviously broken filters
            if "LIKE" in filter_str.upper() and "'%"not in filter_str:
                print(f"[WARN] Malformed filter: {filter_str}")
                filter_str = None
        
        # Use relevance score threshold to filter out irrelevant results
        # Based on testing with 8 recipes:
        #   0.030+: Excellent match (direct query like "mac and cheese" → Mac and Cheese)
        #   0.016-0.020: Weak match (generic queries return all recipes with similar scores)
        #   <0.015: Poor match
        # Setting to 0.025 means: Only show excellent/direct matches, filter weak ones
        min_score = 0.025  # Strict: Only excellent matches
        
        results = engine.search_by_text(
            query, 
            top_k=5 if not show_full else 1, 
            where=filter_str,
            min_score=min_score
        )
        
        if results.empty:
            return "No recipes found matching your criteria. Try different keywords or remove some filters."
        
        response_parts = []
        for _, row in results.iterrows():
            response_parts.append(f"## {row['title']}")
            
            if 'visual_description' in row and row['visual_description']:
                response_parts.append(f"_{row['visual_description']}_")

            calories = row.get("calories_per_serving") if hasattr(row, "get") else None
            level = row.get("calorie_level") if hasattr(row, "get") else ""
            if calories is not None and not (isinstance(calories, float) and calories != calories):
                calories_int = int(round(float(calories)))
                level_part = f" ({level})" if level else ""
                response_parts.append(f"**Calories/serving:** {calories_int}{level_part}")
            
            response_parts.append(f"\n**Ingredients:**\n{row['ingredients']}")
            
            # Show full instructions if requested
            if show_full and 'instructions' in row:
                response_parts.append(f"\n**Instructions:**\n{row['instructions']}")
            
            response_parts.append("\n---")
        
        return "\n".join(response_parts)
    
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        # Retry without filter
        if filter_str:
            return search_with_filter(query, None, show_full) + "\n\n(Note: Filter was invalid, showing unfiltered results)"
        return f"Search error: {str(e)}"


def format_results(results_df, show_full: bool = False) -> str:
    """
    Format search results DataFrame into a readable string.
    
    Args:
        results_df: pandas DataFrame with search results
        show_full: If True, include full instructions
        
    Returns:
        Formatted string with recipe details
    """
    if results_df.empty:
        return "No recipes found matching your criteria."
    
    response_parts = []
    for _, row in results_df.iterrows():
        response_parts.append(f"## {row['title']}")
        
        if 'visual_description' in row and row['visual_description']:
            response_parts.append(f"_{row['visual_description']}_")

        calories = row.get("calories_per_serving") if hasattr(row, "get") else None
        level = row.get("calorie_level") if hasattr(row, "get") else ""
        if calories is not None and not (isinstance(calories, float) and calories != calories):
            calories_int = int(round(float(calories)))
            level_part = f" ({level})" if level else ""
            response_parts.append(f"**Calories/serving:** {calories_int}{level_part}")
        
        response_parts.append(f"\n**Ingredients:**\n{row['ingredients']}")
        
        if show_full and 'instructions' in row:
            response_parts.append(f"\n**Instructions:**\n{row['instructions']}")
        
        response_parts.append("\n---")
    
    return "\n".join(response_parts)


def chat_with_agent(user_message: str, history: list = None, image_file = None) -> str:
    """
    Main chat interface for the agent.
    
    Args:
        user_message: User's query
        history: Chat history (optional, for context)
        image_file: Optional uploaded image file for image-aware search
    
    Returns:
        Agent's response
    """
    if history is None:
        history = []
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history (handle Streamlit's format: {'role': 'user'/'assistant', 'content': '...'})
    # Exclude the last message if it's the current user message (to avoid duplicates)
    history_to_add = history[:-1] if history else []
    for h in history_to_add:
        if 'role' in h and 'content' in h:
            messages.append({"role": h['role'], "content": h['content']})
    
    # Generate image context if image is provided
    image_context = None
    if image_file:
        try:
            print("[AGENT] Generating image context...")
            image_context = engine._caption_image_groq(image_file)
            # Reset file pointer for later use
            if hasattr(image_file, 'seek'):
                image_file.seek(0)
        except Exception as e:
            print(f"[WARN] Failed to get image context: {e}")
            image_context = "Unable to analyze image"
    
    # Build user message with image context
    if image_context:
        enhanced_message = f"IMAGE_CONTEXT: {image_context}\n\nUser Query: {user_message}"
    else:
        enhanced_message = user_message
    
    # Add current message
    messages.append({"role": "user", "content": enhanced_message})
    
    try:
        # Call LLM
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        
        llm_output = response.choices[0].message.content.strip()
        print(f"[LLM Output]\n```\n{llm_output}\n```\n")
        
        # Parse output
        parsed = parse_agent_output(llm_output)
        print(f"[Parsed] Query: {parsed['query']}, Filter: {parsed['filter']}, Full: {parsed['full']}")
        
        # Handle clarification
        if parsed['query'] and parsed['query'].upper() == 'CLARIFY':
            return parsed['response'] or "Could you please be more specific about what you're looking for?"
        
        # Handle knowledge/reasoning questions (no search needed)
        if parsed['query'] and parsed['query'].upper() == 'KNOWLEDGE':
            return parsed['response'] or "I can help with that question."
        
        # Handle IMAGE_SEARCH - use hybrid image search
        if parsed['query'] and parsed['query'].upper().startswith('IMAGE_SEARCH'):
            if image_file:
                print("[AGENT] Executing hybrid image search...")
                # Check for text modification (e.g., "IMAGE_SEARCH | strawberry cake")
                query_parts = parsed['query'].split('|')
                text_modification = query_parts[1].strip() if len(query_parts) > 1 else None
                
                if text_modification:
                    # Composed query: search by modified text
                    print(f"[AGENT] Composed query with modification: {text_modification}")
                    search_results = search_with_filter(text_modification, parsed['filter'], parsed['full'])
                else:
                    # Pure image search
                    image_file.seek(0) if hasattr(image_file, 'seek') else None
                    results_df = engine.search_by_image_hybrid(image_file, top_k=5)
                    search_results = format_results(results_df, parsed['full'])
                
                if parsed['response']:
                    return f"{parsed['response']}\n\n{search_results}"
                return search_results
            else:
                return "I need an image to search. Please upload a food photo first."
            
        # Execute text search if we have a query
        if parsed['query']:
            search_results = search_with_filter(parsed['query'], parsed['filter'], parsed['full'])
            
            # Combine agent response with search results
            if parsed['response']:
                return f"{parsed['response']}\n\n{search_results}"
            return search_results
        
        # Fallback if no query
        return parsed['response'] or "I couldn't understand your request. Could you rephrase?"
        
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Interactive test
    print("Culinary Compass AI - Recipe Search Agent")
    print("=" * 50)
    print("Try queries like:")
    print("  - 'I want something with chicken but no dairy'")
    print("  - 'vegetarian pasta'")
    print("  - 'How do I make mac and cheese?'")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        response = chat_with_agent(user_input)
        print(f"\nAgent: {response}")
