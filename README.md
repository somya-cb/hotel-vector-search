# Hotel Vector Search

A hybrid semantic + keyword hotel search app powered by **Couchbase Vector Search** and **Streamlit**. Users describe their ideal stay in plain English, and the app uses vector embeddings alongside structured filters to return the most relevant hotels.

---

## Features

- **Semantic search** — Natural language queries are embedded using `sentence-transformers/all-MiniLM-L6-v2` and matched against pre-computed hotel description embeddings stored in Couchbase.
- **Hybrid filtering** — Combine semantic results with structured FTS filters: city, state, free parking, free breakfast, vacancy, WiFi, and pet-friendly.
- **NLP query parsing** — Keywords like "wifi", "pet friendly", and "free parking" are automatically extracted from the query and applied as filters.
- **Streamlit UI** — Clean, interactive web interface with dropdowns, checkboxes, and a search button.

---

## Project Structure

```
hotel-vector-search/
├── hotel_search_app.py      # Main Streamlit app
├── embedding_loader.py      # One-time script to generate & store embeddings in Couchbase
├── filtered_vectorstore.py  # Custom LangChain vector store with prefilter support
├── config.py                # Loads configuration from environment variables
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
└── .env                     # Your local credentials (gitignored, never committed)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/hotel-vector-search.git
cd hotel-vector-search
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your Couchbase credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
COUCHBASE_CONN_STR=couchbases://cb.<your-cluster-id>.cloud.couchbase.com
COUCHBASE_USERNAME=your-username
COUCHBASE_PASSWORD=your-password
BUCKET_NAME=travel-sample
SCOPE_NAME=inventory
COLLECTION_NAME=hotel
VECTOR_INDEX_NAME=vector_idx
VECTOR_FIELD=description_embedding
```

### 4. Generate embeddings (first-time setup)

Run this once to embed hotel descriptions and store them back in Couchbase:

```bash
python embedding_loader.py
```

### 5. Run the app

```bash
streamlit run hotel_search_app.py
```

---

## Requirements

- Python 3.10+
- A [Couchbase Capella](https://cloud.couchbase.com/) cluster with the `travel-sample` bucket loaded
- A vector search index named `vector_idx` on the `description_embedding` field

---

## How It Works

1. The user types a natural language query (e.g. *"cozy hotel near the beach with free wifi"*).
2. Keywords like "wifi" are extracted and mapped to boolean FTS filters.
3. The remaining semantic query is encoded into a 384-dim vector.
4. Couchbase performs a **pre-filtered vector search**: FTS filters narrow the candidate set, then vector similarity ranks the results.
5. Full hotel documents are fetched and displayed with amenities, location, price, and vacancy.

<img width="1904" height="766" alt="image" src="https://github.com/user-attachments/assets/4757de78-ec67-4245-b575-dcfc4edf2a68" />

### Results
<img width="1805" height="1014" alt="image" src="https://github.com/user-attachments/assets/119f1059-1e11-4d3c-aee4-bc1afc3fc2ed" />

