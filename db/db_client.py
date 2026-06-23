from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv
from utils import COLLECTION_NAME
from retrieval.hybrid_search import SPARSE_VECTOR_NAME

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CONNECTION_STRING"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

try:
    collection_info = qdrant_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print(f"Collection '{COLLECTION_NAME}' already exists")


except Exception:

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams()
        },
    )

    print(
        f"Created hybrid collection '{COLLECTION_NAME}'"
    )



