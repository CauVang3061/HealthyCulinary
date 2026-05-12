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

SYSTEM_PROMPT = """You are Culinary Compass AI, a helpful recipe and nutrition assistant.
 
When a user sends a message, analyze their intent and respond in EXACTLY this format:
 
INTENT: <one of: SEARCH | CALORIE_CHECK | IMAGE_SEARCH | KNOWLEDGE | CLARIFY>
QUERY: <search query, food description, or special keyword>
FILTER: <SQL filter or NONE>
FULL: <YES or NO>
RESPONSE: <your conversational reply to the user>
 
--- INTENT RULES ---
 
INTENT = SEARCH
  Use when user wants to find recipes, e.g. "recipe for chicken", "vegetarian pasta", "something with cheese"
  QUERY = search keywords
  FILTER = SQL filter like: calories_per_serving < 500, array_has_any(tags, ['Vegetarian'])
  FULL = YES only if user asks "how to make" or wants full instructions
  RESPONSE = brief intro before results
 
INTENT = CALORIE_CHECK
  Use when user wants to:
    - Know how many calories in their meal
    - Evaluate if a meal is healthy
    - Get suggestions to reduce calories
    - Log what they ate
  Keywords: "calo", "calories", "I want to eat", "meal", "healthy",
            "change", "I ate", "how many calories", "is this healthy"
  QUERY = exact meal description (copy from user, or describe image content)
  FILTER = meal type: "breakfast" | "lunch" | "dinner" | "snack" | "general"
           Default to "general" if not specified.
           Look for hints: "breakfast/lunch/dinner"
  FULL = NO (not used for calorie check)
  RESPONSE = brief acknowledgment, e.g. "Let me help you check the calories for that meal!"
 
INTENT = IMAGE_SEARCH
  Use when user uploads an image and wants recipe search (not calorie check).
  QUERY = IMAGE_SEARCH or IMAGE_SEARCH | <text modifier>
  FILTER = SQL filter or NONE
  FULL = YES/NO
 
INTENT = KNOWLEDGE
  Use for general food/nutrition questions that don't need search.
  QUERY = KNOWLEDGE
  RESPONSE = direct answer
 
INTENT = CLARIFY
  Use when request is too vague to act on.
  QUERY = CLARIFY
  RESPONSE = specific clarifying question
 
--- CALORIE_CHECK EXAMPLES ---
 
User: "I had a large bowl of mac and cheese for dinner, is that too much?"
INTENT: CALORIE_CHECK
QUERY: large bowl of mac and cheese
FILTER: dinner
FULL: NO
RESPONSE: Let me check the calories for your dinner!
 
--- IMAGE + CALORIE EXAMPLE ---
 
If IMAGE_CONTEXT is provided and user asks about calories:
INTENT: CALORIE_CHECK
QUERY: <use the IMAGE_CONTEXT description as the meal description>
FILTER: general
 
--- IMPORTANT ---
- Always output all 5 fields (INTENT, QUERY, FILTER, FULL, RESPONSE).
- RESPONSE must always be in the same language as the user's message.
- For CALORIE_CHECK, QUERY must preserve the meal description exactly — do not paraphrase.
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
    
    if not result["intent"]:
        q = (result["query"] or "").upper()
        if q == "CLARIFY":
            result["intent"] = "CLARIFY"
        elif q == "KNOWLEDGE":
            result["intent"] = "KNOWLEDGE"
        elif q.startswith("IMAGE_SEARCH"):
            result["intent"] = "IMAGE_SEARCH"
        else:
            result["intent"] = "SEARCH"

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

def handle_calories_check(meal_description: str, meal_type: str) -> str:
    """
    Run calorie pipeline:
      MealParser → CalorieAssessor → SuggestionEngine
 
    Args:
        meal_description: Describe the meal in detail
        meal_type:        "breakfast" | "lunch" | "dinner" | "snack" | "general"
 
    Returns:
        Formatted markdown string
    """
    try:
        from meal_parser      import MealParser
        from calories_assessor import CalorieAssessor, format_assessment_report
        from suggest_engine import SuggestionEngine
 
        parser    = MealParser()
        assessor  = CalorieAssessor()
        suggester = SuggestionEngine()
 
        # 1. Parse
        print(f"[CALORIE PIPELINE] Parsing: '{meal_description[:60]}...'")
        items = parser.parse_from_text(meal_description)
 
        if not items:
            return (
                "I couldn't identify the specific dish in your description. "
                "Could you list it more clearly? "
                "For example: '1 bowl of beef noodles, 2 spring rolls'"
            )
 
        # 2. Assess
        result = assessor.assess(items, meal_type=meal_type)
        report = format_assessment_report(result)
 
        # 3. Suggest
        if result["needs_suggestion"]:
            suggestions = suggester.suggest(result)
            suggestion_text = suggester.format_suggestions(suggestions)
            return f"{report}\n\n---\n\n{suggestion_text}"
 
        return report
 
    except Exception as e:
        print(f"[CALORIE PIPELINE][ERROR] {e}")
        return f"Sorry, there was an error processing your request: {str(e)}"

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
    print("  - 'How should I make a healthy dinner?'")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        response = chat_with_agent(user_input)
        print(f"\nAgent: {response}")
