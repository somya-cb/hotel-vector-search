from langchain_couchbase.vectorstores import CouchbaseVectorStore
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest
from couchbase.options import SearchOptions
from typing import List, Optional, Any, Tuple
from langchain_core.documents import Document

class FilteredCouchbaseVectorStore(CouchbaseVectorStore):
    def __init__(self, *args, embedding_key: str, text_key: str, **kwargs):
        super().__init__(embedding_key=embedding_key, text_key=text_key, *args, **kwargs)
        self.embedding_key = embedding_key
        self.text_key = text_key


    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        search_options: Optional[dict] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        vector_query = VectorQuery.of(embedding).with_field(self.embedding_key).with_k(k)

        search_req = SearchRequest.create(self.index_name)
        search_req.with_vector_search(VectorSearch.from_vector_query(vector_query))

        # Add prefilter
        if search_options and "query" in search_options:
            search_req.query(search_options["query"])

        opts = SearchOptions()
        result = self.cluster.search(search_req, opts)

        docs = []
        for row in result.rows():
            doc = self._row_to_document(row)
            docs.append((doc, row.score))

        return docs
