import os
from dotenv import load_dotenv

load_dotenv()

COUCHBASE_CONN_STR = os.getenv("COUCHBASE_CONN_STR")
COUCHBASE_USERNAME = os.getenv("COUCHBASE_USERNAME")
COUCHBASE_PASSWORD = os.getenv("COUCHBASE_PASSWORD")
BUCKET_NAME = os.getenv("BUCKET_NAME", "travel-sample")
SCOPE_NAME = os.getenv("SCOPE_NAME", "inventory")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hotel")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_idx")
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "description_embedding")
