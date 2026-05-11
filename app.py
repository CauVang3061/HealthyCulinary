import streamlit as st
import os
import ast
import re
from dotenv import load_dotenv
from PIL import Image
from search_engine import RecipeSearchEngine

load_dotenv()


st.set_page_config(page_title="Culinary Compass", page_icon="🧭", layout="wide")
IMAGES_DIR = "Food Images"

states = {
    'view': 'home',
    'selected_recipe': None,
    'search_results': None,
    'search_type': None,
    'chat_history': [],
    'active_chat': False,
    'main_query': "",
    'fridge_input': "",
    'is_strict': False,
    'pantry_radio_index': 0
}

for key, default in states.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def get_search_engine():
    return RecipeSearchEngine()

engine = get_search_engine()

def format_instruction_list(text):
    if not text or str(text).lower() in ['nan', 'none', '']:
        return ""
    items = []
    if isinstance(text, list):
        items = text
    else:
        try:
            parsed = ast.literal_eval(str(text))
            items = parsed if isinstance(parsed, list) else [str(parsed)]
        except:
            items = str(text).split('\n')
    cleaned = [f"* {re.sub(r'^[•\-\*]\s*', '', str(item)).strip()}" for item in items if str(item).strip()]
    return "\n".join(cleaned)

def get_recipe_image(image_name):
    if not image_name or str(image_name).lower() == 'nan': return None
    if not str(image_name).lower().endswith(".jpg"): image_name = f"{image_name}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_name)
    return Image.open(image_path) if os.path.exists(image_path) else None

def render_search_ui():
    st.title("Culinary Compass 🧭")
    
    query = st.text_input(
        "Search by name, cuisine, or craving...", 
        value=st.session_state.main_query,
        key="query_widget"
    )
    st.session_state.main_query = query

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🖼️ Images Search", use_container_width=True):
            st.session_state.search_type = "image"
    with c2:
        if st.button("🧊 What's in My Fridge?", use_container_width=True):
            st.session_state.search_type = "fridge"
    with c3:
        if st.button("🤖 Search Smart with AI", use_container_width=True):
            st.session_state.view = 'ai_search'
            st.rerun()

    if query and query != st.session_state.get('last_query'):
        with st.spinner("Searching..."):
            st.session_state.search_results = engine.search_by_text(query)
            st.session_state.search_type = "text"
            st.session_state.last_query = query

    if st.session_state.search_type == "image":
        uploaded = st.file_uploader("Upload food photo", type=["jpg", "png", "jpeg"])
        if uploaded:
            st.session_state.search_results = engine.search_by_image_hybrid(uploaded)
    
    if st.session_state.search_type == "fridge":
        ing = st.text_area(
            "Ingredients (comma separated):", 
            value=st.session_state.fridge_input,
            placeholder="Eggs, Milk, Flour...",
            key="fridge_widget"
        )
        st.session_state.fridge_input = ing

        options = ("Strict (I can make this now)", "Flexible (Suggestions with missing items)")
        pantry_mode = st.radio(
            "Filter Mode:",
            options,
            index=st.session_state.pantry_radio_index,
            horizontal=True,
            key="radio_widget"
        )
        st.session_state.is_strict = pantry_mode.startswith("Strict")
        st.session_state.pantry_radio_index = options.index(pantry_mode)
        
        if st.button("Find Recipes"):
            with st.spinner("Checking the pantry..."):
                st.session_state.search_results = engine.search_by_ingredients(ing, strict=st.session_state.is_strict)

def render_results_grid():
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        if results.empty:
            st.info("No recipes found matching those criteria.")
            return

        st.divider()
        cols = st.columns(3)
        for i, (idx, row) in enumerate(results.iterrows()):
            with cols[i % 3]:
                with st.container(border=True, height=500):
                    img = get_recipe_image(row['image_name'])
                    if img: st.image(img, use_container_width=True)
                    
                    if st.button(row['title'], key=f"btn_{idx}", use_container_width=True):
                        st.session_state.selected_recipe = row
                        st.session_state.view = 'detail'
                        st.rerun()
                    
                    calories = row.get("calories_per_serving")
                    if calories and str(calories).lower() != 'nan':
                        st.caption(f"🔥 {int(float(calories))} Cal/serving")
                    
                    if st.session_state.search_type == "fridge":
                        missing = row.get('missing_ingredients', [])
                        if not missing:
                            st.success("✅ Ready to cook!")
                        else:
                            st.warning(f"⚠️ Missing {len(missing)} items")

def render_recipe_blog():
    recipe = st.session_state.selected_recipe
    
    col_img, col_info = st.columns([1, 2])
    with col_img:
        img = get_recipe_image(recipe['image_name'])
        if img: st.image(img, width=350)
    
    with col_info:
        st.title(recipe['title'])
        calories = recipe.get("calories_per_serving")
        if calories and str(calories).lower() != 'nan':
            st.write(f"**Calories per serving:** {int(float(calories))}")
        
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            chat_click = st.button(f"💬 Chat about this recipe", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("⬅️ Back to Search", use_container_width=True):
                st.session_state.view = 'home'
                st.rerun()

    if chat_click or st.session_state.active_chat:
        st.session_state.active_chat = True
        st.divider()
        st.subheader(f"Conversation about {recipe['title']}")
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Ask a question about this recipe..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            response = f"I'm assisting you with the {recipe['title']}. How can I help with the steps?"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"): st.markdown(response)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ingredients")
        missing = recipe.get('missing_ingredients', [])
        if st.session_state.search_type == "fridge" and missing:
            st.error(f"**Missing:** {', '.join(missing)}")
        st.markdown(format_instruction_list(recipe['ingredients']))
    with c2:
        st.subheader("Instructions")
        st.markdown(format_instruction_list(recipe['instructions']))

def render_ai_smart_search():
    if st.button("⬅️ Back to Main UI"):
        st.session_state.view = 'home'
        st.rerun()
    st.header("Culinary Compass AI 🤖")
    if "agent_history" not in st.session_state: st.session_state.agent_history = []
    uploaded_image = st.file_uploader("📸 Upload food image (optional)", type=["jpg", "png", "jpeg"])
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("How can I help today?"):
        st.session_state.agent_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.spinner("Thinking..."):
            try:
                from agent import chat_with_agent
                res = chat_with_agent(prompt, st.session_state.agent_history, image_file=uploaded_image)
                st.session_state.agent_history.append({"role": "assistant", "content": res})
                with st.chat_message("assistant"): st.markdown(res)
            except Exception as e: st.error(f"Agent Error: {e}")

if st.session_state.view == 'home':
    render_search_ui()
    render_results_grid()
elif st.session_state.view == 'detail':
    render_recipe_blog()
elif st.session_state.view == 'ai_search':
    render_ai_smart_search()