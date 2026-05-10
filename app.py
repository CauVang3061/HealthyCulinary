import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
from PIL import Image
from search_engine import RecipeSearchEngine

# --- Configuration ---
st.set_page_config(page_title="Culinary Compass", page_icon="🧭", layout="wide")
IMAGES_DIR = "Food Images"

# --- Initialization ---
@st.cache_resource
def get_search_engine_v3():
    # Force cache reload after logic update
    return RecipeSearchEngine()

engine = get_search_engine_v3()

# --- UI Layout ---
st.sidebar.title("Culinary Compass 🧭")
search_mode = st.sidebar.radio("Search Mode", ["Search by Name", "Search by Image", "What's in my Fridge?", "AI Smart Search 🤖"])

st.title("Culinary Compass 🧭")
st.markdown("### Find your next delicious meal!")

# --- Helper Functions ---
def format_calorie_line(row) -> str:
    calories = row.get("calories_per_serving", None)
    level = row.get("calorie_level", "")
    servings = row.get("servings", None)

    if calories is None:
        return ""
    if isinstance(calories, float) and calories != calories:
        return ""

    calories_int = int(round(float(calories)))
    level_part = f" ({level})" if level else ""
    servings_part = f" | Servings: {int(servings)}" if servings and str(servings).isdigit() else ""
    return f"Calories/serving: {calories_int}{level_part}{servings_part}"

def display_results(results_df):
    if results_df.empty:
        st.info("No recipes found matching your criteria.")
        return

    # Create a grid
    cols = st.columns(3)
    for i, (idx, row) in enumerate(results_df.iterrows()):
        with cols[i % 3]:
            # Load image
            image_name = row['image_name']
            if not image_name.lower().endswith(".jpg"):
                image_name += ".jpg"
            image_path = os.path.join(IMAGES_DIR, image_name)
            
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
            else:
                st.warning(f"Image not found: {image_name}")

            st.subheader(row['title'])

            calorie_line = format_calorie_line(row)
            if calorie_line:
                st.caption(calorie_line)
            
            with st.expander("View Instructions"):
                st.write("**Ingredients:**")
                st.write(row['ingredients'])
                st.write("**Instructions:**")
                st.write(row['instructions'])
            
            # Chatbot feature
            if st.button(f"Chat with {row['title']}", key=f"chat_{idx}"):
                st.session_state['active_recipe'] = row['title']
                st.session_state['chat_history'] = []
                st.rerun()

# --- Chat Interface (Overlay or separate section) ---
if 'active_recipe' in st.session_state:
    st.markdown("---")
    st.subheader(f"Chatting about: {st.session_state['active_recipe']}")
    if st.button("Close Chat"):
        del st.session_state['active_recipe']
        st.rerun()
    else:
        # Display chat history
        messages = st.session_state.get('chat_history', [])
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about this recipe..."):
            # Add user message
            st.session_state['chat_history'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response (Placeholder)
            response = f"That's a great question about {st.session_state['active_recipe']}! As an AI chef, I'd say: Use your best judgment and taste as you go."
            
            st.session_state['chat_history'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# --- Search Modes ---
if 'active_recipe' not in st.session_state: # Only show search if not chatting
    if search_mode == "Search by Name":
        st.header("Search by Name")
        query = st.text_input("Enter recipe name or description:")
        if query:
            with st.spinner("Searching..."):
                results = engine.search_by_text(query)
            display_results(results)

    elif search_mode == "Search by Image":
        st.header("Visual Search")
        uploaded_file = st.file_uploader("Upload a food photo", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            with st.spinner("Analyzing image with AI (visual + semantic)..."):
                # Use hybrid search: CLIP + LLM captioning
                results = engine.search_by_image_hybrid(uploaded_file)
            st.subheader("Best Matching Recipes")
            display_results(results)

    elif search_mode == "What's in my Fridge?":
        st.header("Pantry Search")
        ingredients = st.text_area("Enter ingredients (comma separated):", "Eggs, Milk")
        
        # New: Search Mode Toggle
        pantry_mode = st.radio(
            "Filter Mode:",
            ("Strict (I can make this now)", "Flexible (What can I make with more items?)"),
            help="Strict: Recipes using ONLY what you have.\nFlexible: Recipes using what you have + suggestions."
        )
        
        is_strict = pantry_mode.startswith("Strict")

        if st.button("Find Recipes"):
            with st.spinner("Checking the pantry..."):
                results = engine.search_by_ingredients(ingredients, strict=is_strict)
            
            if not results.empty:
                # Custom display for pantry results to show missing ingredients
                cols = st.columns(3)
                for i, (idx, row) in enumerate(results.iterrows()):
                    with cols[i % 3]:
                        # Load image
                        image_name = row['image_name']
                        if not image_name.lower().endswith(".jpg"):
                            image_name += ".jpg"

                        image_path = os.path.join(IMAGES_DIR, image_name)
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            st.image(image, use_container_width=True)
                        else:
                            st.warning(f"Image not found: {image_name}")

                        st.subheader(row['title'])

                        calorie_line = format_calorie_line(row)
                        if calorie_line:
                            st.caption(calorie_line)
                        
                        # Show missing ingredients if flexible mode
                        if not is_strict and 'missing_ingredients' in row and row['missing_ingredients']:
                            st.error(f"Missing: {', '.join(row['missing_ingredients'])}")
                        elif not is_strict:
                            st.success("You have everything!")
                        elif is_strict:
                            st.success("You can make this!")

                        with st.expander("View Instructions"):
                            st.write("**Ingredients:**")
                            st.write(row['ingredients'])
                            st.write("**Instructions:**")
                            st.write(row['instructions'])
                        
                        if st.button(f"Chat with {row['title']}", key=f"chat_{idx}"):
                            st.session_state['active_recipe'] = row['title']
                            st.session_state['chat_history'] = []
                            st.rerun()
            else:
                 st.info("No recipes found. Try adding more ingredients or switching to Flexible mode.")

    elif search_mode == "AI Smart Search 🤖":
        st.header("Culinary Compass AI 🤖")
        st.markdown("*Ask me anything! Upload a food photo, find recipes, or handle complex constraints.*")

        if "agent_history" not in st.session_state:
            st.session_state.agent_history = []
        if "agent_image" not in st.session_state:
            st.session_state.agent_image = None

        # Image upload section
        col1, col2 = st.columns([1, 3])
        with col1:
            uploaded_image = st.file_uploader("📸 Upload food image", type=["jpg", "png", "jpeg"], key="agent_img_uploader")
            if uploaded_image:
                st.session_state.agent_image = uploaded_image
                st.image(uploaded_image, caption="Uploaded", width=150)
            elif st.session_state.agent_image:
                st.image(st.session_state.agent_image, caption="Current image", width=150)
                if st.button("Clear image"):
                    st.session_state.agent_image = None
                    st.rerun()
        
        with col2:
            # Display history
            for msg in st.session_state.agent_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("How can I help you today?"):
            st.session_state.agent_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                try:
                    from agent import chat_with_agent
                    
                    # Pass image if available
                    response_text = chat_with_agent(
                        prompt, 
                        st.session_state.agent_history,
                        image_file=st.session_state.agent_image
                    )
                    
                    st.session_state.agent_history.append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                except Exception as e:
                    st.error(f"Agent Error: {e}")

