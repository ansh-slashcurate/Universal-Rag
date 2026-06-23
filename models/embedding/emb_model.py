from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from fastembed import SparseTextEmbedding

# embedding_model = SentenceTransformer(
#     "ibm-granite/granite-embedding-107m-multilingual",
#     token=os.getenv("HF_TOKEN")
# )
# embedding_model.save("./local_models/embedding_model")

# reranker = CrossEncoder(
#      "BAAI/bge-reranker-base", 
#      token=os.getenv("HF_TOKEN")
# )
# reranker.save("./local_models/reranker")


embedding_model = None
reranker = None
sparse_embedding = None

class ModelManager:

    def __init__(self):

        self.embedding_model = None
        self.reranker = None
        self.sparse_embedding = None

    def load_models(self):

        print(f"Loading models")

        self.embedding_model = SentenceTransformer(
        "./local_models/embedding_model"
        )

        self.reranker = CrossEncoder(
        "./local_models/reranker"
        )

        self.sparse_embedding = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

        # self.sparse_embedding.save("./local_models/sparse_embedding")

        print(f"Models loaded")

model_manager = ModelManager()






