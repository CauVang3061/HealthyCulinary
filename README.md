# 🧭 HealthyCulinary (Culinary Compass)

**HealthyCulinary** is a multimodal AI-powered recipe search engine. It helps you find the perfect dish using text descriptions, food images, or by checking what's in your pantry. Now enhanced with LLM-generated visual descriptions and smart tags for improved search relevance.

## 🌟 Features

- **Search by Name**: Find recipes by title or description using semantic search (MiniLM).
- **Visual Search**: Upload a photo of a dish to find visually similar recipes (OpenAI CLIP).
- **Pantry Search (Smart)**:
  - **Strict Mode**: Find recipes you can make *right now* with your current ingredients.
  - **Flexible Mode**: Find recipes you can *partially* make, prioritizing the ones closest to completion.
- **AI Chat**: Chat with the "Culinary Compass AI" to get recipe recommendations with natural language queries (powered by Groq Llama).
- **LLM Enrichment**: Recipes are enriched with AI-generated visual descriptions and tags (cuisine, diet, course, vibe) for better search results.
- **Calorie Enrichment**: Optional per-serving calorie estimates with Low/Medium/High labels (Spoonacular API).

## 🛠️ Setup

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/CauVang3061/CaloriesEvaluation.git
cd CaloriesEvaluation
pip install -r requirements.txt
```

### 3. Data Preparation
Place your dataset in the root directory:
1. **CSV File**: `Food Ingredients and Recipe Dataset with Image Name Mapping.csv`
2. **Images Folder**: `Food Images/` (containing recipe JPG/PNG files)

## 🔄 Two-Phase Ingestion Pipeline

### Phase 1: Local Embeddings (Fast, Free)
```bash
python ingest.py
```
- Reads CSV and cleans data
- Generates text embeddings (MiniLM)
- Generates image embeddings (CLIP)
- Saves to LanceDB

### Phase 2: LLM Enrichment (Resumable, API)
```bash
python enrich_recipes.py           # Enrich all remaining
python enrich_recipes.py --limit 50  # Enrich 50 at a time
python enrich_recipes.py --status    # Check progress
```
- Uses Groq's Llama 4 Scout Vision API
- Generates `visual_description` and `tags` for each recipe
- **Resumable**: Run multiple times to complete large datasets
- Progress is saved after each recipe

### Optional: Calorie Enrichment (Resumable, API)
```bash
python enrich_calories.py
python enrich_calories.py --limit 50
python enrich_calories.py --status
```
- Uses Spoonacular's `analyzeRecipe` API
- Stores `calories_per_serving`, `calorie_level`, and `servings`
- Requires `SPOONACULAR_API_KEY` in your `.env`

## 🚀 Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

Or on Windows, double-click: `run_app.bat`

## 🧠 Technologies

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Vector DB | LanceDB |
| Text Embeddings | `all-MiniLM-L6-v2` |
| Image Embeddings | `openai/clip-vit-base-patch32` |
| AI Chat | Groq API (Llama 3.3 70B) |
| Vision Enrichment | Groq API (Llama 4 Scout) |

## 📊 Database Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique recipe ID |
| `title` | str | Recipe name |
| `ingredients` | str | List of ingredients |
| `instructions` | str | Cooking instructions |
| `image_name` | str | Image filename |
| `visual_description` | str | AI-generated appearance description |
| `tags` | list[str] | AI-generated tags (cuisine, diet, course, vibe) |
| `servings` | int | Estimated serving count |
| `calories_per_serving` | float | Estimated calories per serving |
| `calorie_level` | str | Low/Medium/High calorie label |
| `text_vector` | vector | Text embedding (384 dims) |
| `image_vector` | vector | Image embedding (512 dims) |

## 📝 License

MIT License
