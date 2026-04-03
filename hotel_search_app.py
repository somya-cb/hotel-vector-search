import re
import streamlit as st
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions, SearchOptions
from couchbase.search import TermQuery, BooleanFieldQuery, ConjunctionQuery, MatchQuery
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest

from config import (
    COUCHBASE_CONN_STR,
    COUCHBASE_USERNAME,
    COUCHBASE_PASSWORD,
    BUCKET_NAME,
    SCOPE_NAME,
    COLLECTION_NAME,
    VECTOR_INDEX_NAME,
    VECTOR_FIELD,
)

st.set_page_config(page_title="Hotel Search", layout="wide")
st.title("🏨 Hotel Search")

# ── Caching ────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def connect_cluster():
    cluster = Cluster(
        COUCHBASE_CONN_STR,
        ClusterOptions(PasswordAuthenticator(COUCHBASE_USERNAME, COUCHBASE_PASSWORD))
    )
    cluster.wait_until_ready(timeout=timedelta(seconds=10))
    return cluster

embedding_model = load_embedding_model()
cluster = connect_cluster()
bucket = cluster.bucket(BUCKET_NAME)
scope = bucket.scope(SCOPE_NAME)
collection = scope.collection(COLLECTION_NAME)

# ── Dropdown Options ───────────────────────────────────────
@st.cache_data(ttl=3600)
def get_available_values(field: str):
    sql = f"""
        SELECT DISTINCT `{field}`
        FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
        WHERE `{field}` IS NOT NULL AND `{field}` != ""
        ORDER BY `{field}`
    """
    result = cluster.query(sql, QueryOptions(metrics=False))
    return [""] + [row[field] for row in result if row[field]]

# ── NLP Query Parsing ──────────────────────────────────────
def parse_query_components(user_query: str):
    text = user_query.lower()
    keyword_map = {
        "wifi": "free_internet", "internet": "free_internet", "wi-fi": "free_internet",
        "pet friendly": "pets_ok", "pets allowed": "pets_ok", "pet-friendly": "pets_ok",
        "free parking": "free_parking", "parking": "free_parking",
        "breakfast": "free_breakfast", "free breakfast": "free_breakfast",
    }
    filters = {}
    cleaned = user_query
    for kw, field in keyword_map.items():
        if kw in text:
            filters[field] = True
            cleaned = cleaned.replace(kw, " ")
    semantic_query = re.sub(r'\b(with|and|the|a|an|in|at|for|to|from|near)\b', ' ', cleaned)
    semantic_query = re.sub(r'\s+', ' ', semantic_query).strip()
    return semantic_query, filters

# ── FTS Filter Builder ─────────────────────────────────────
def build_fts_filter(city, state, ui_flags, parsed_flags):
    clauses = []

    if city and city.strip():
        clauses.append(MatchQuery(city.strip().lower(), field="city"))
    if state and state.strip():
        clauses.append(MatchQuery(state.strip().lower(), field="state"))

    def add_bool(field, flag):
        if flag:
            clauses.append(BooleanFieldQuery(True, field=field))

    add_bool("free_parking", ui_flags.get("free_parking") or parsed_flags.get("free_parking"))
    add_bool("free_breakfast", ui_flags.get("free_breakfast") or parsed_flags.get("free_breakfast"))
    add_bool("vacancy", ui_flags.get("vacancy"))
    add_bool("free_internet", parsed_flags.get("free_internet"))
    add_bool("pets_ok", parsed_flags.get("pets_ok"))

    return ConjunctionQuery(*clauses) if clauses else None

# ── Hybrid Search ──────────────────────────────────────────
def hybrid_search(query_text, k, fts_filter):
    # Create embedding
    embedding = embedding_model.encode(query_text).tolist()
    
    # Build vector query with prefilter
    vector_query = VectorQuery(
        field_name=VECTOR_FIELD,
        vector=embedding,
        num_candidates=k,
        prefilter=fts_filter
    )
    
    # Create search request
    search_req = SearchRequest.create(VectorSearch.from_vector_query(vector_query))
    
    # Execute search
    results = scope.search(
        VECTOR_INDEX_NAME,
        search_req,
        SearchOptions(limit=k, fields=["*"])
    )
    
    # Fetch full documents
    enriched_results = []
    for row in results.rows():
        try:
            full_doc = collection.get(row.id).content_as[dict]
            enriched_results.append((full_doc, row.score))
        except Exception:
            # Fallback to fields if get fails
            enriched_results.append((row.fields or {}, row.score))
    
    return enriched_results

# ── UI Inputs ──────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    user_query = st.text_input("Describe your stay:", placeholder="e.g. luxury hotel with wifi", value="luxury hotel")
with col2:
    cities = get_available_values("city")
    states = get_available_values("state")
    city = st.selectbox("City", cities)
    state = st.selectbox("State", states)
    col2a, col2b = st.columns(2)
    with col2a:
        free_parking = st.checkbox("Free Parking")
        vacancy = st.checkbox("Vacancy")
    with col2b:
        free_breakfast = st.checkbox("Free Breakfast")

# ── Run Search ─────────────────────────────────────────────
if st.button("🔍 Search"):
    with st.spinner("Searching..."):
        semantic_query, parsed_flags = parse_query_components(user_query)
        fts_filter = build_fts_filter(
            city, state,
            ui_flags={"free_parking": free_parking, "free_breakfast": free_breakfast, "vacancy": vacancy},
            parsed_flags=parsed_flags
        )

        hits = hybrid_search(semantic_query, k=10, fts_filter=fts_filter)

        # Fallback without filters
        if not hits and fts_filter:
            st.write("🔄 **Trying without filters...**")
            hits = hybrid_search(semantic_query, k=10, fts_filter=None)

        if hits:
            st.success(f"✅ Found {len(hits)} hotels")
            for i, (doc, score) in enumerate(hits, 1):
                st.markdown(f"### {i}. {doc.get('name', 'Unnamed')}")
                st.markdown(f"_{doc.get('description', '')}_")
                st.markdown(f"- 📍 **Location**: {doc.get('city', '—')}, {doc.get('country', '—')}")
                st.markdown(f"- 💰 **Price**: {doc.get('price', 'Not specified')}")
                st.markdown(f"- ⭐ **Score**: {score:.3f}")
                
                vacancy_text = "✅ Available" if doc.get("vacancy") else "❌ Not Available"
                st.markdown(f"- 🛏️ **Vacancy**: {vacancy_text}")
                
                amenities = []
                if doc.get("free_parking"): amenities.append("🅿️ Parking")
                if doc.get("free_breakfast"): amenities.append("🥐 Breakfast")
                if doc.get("free_internet"): amenities.append("📶 WiFi")
                if doc.get("pets_ok"): amenities.append("🐕 Pet Friendly")
                if amenities:
                    st.markdown(f"- 🎯 **Amenities**: {', '.join(amenities)}")
        else:
            st.info("No results. Try relaxing filters or broadening your query.")