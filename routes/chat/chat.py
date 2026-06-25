import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from models.embedding.emb_model import model_manager
from models.llm.llm import watsonx_llm, google_llm
from db.db_client import qdrant_client
from llama_index.core.llms import ChatMessage
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch,FusionQuery, Fusion, SparseVector
from retrieval.hybrid_search import SPARSE_VECTOR_NAME
from db.redis import upStash_chatStore
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from lib.chat_memory import get_chat_memory
from lib.session import generate_session_id
import uuid
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from utils import COLLECTION_NAME
import time

import hashlib

router = APIRouter()

SEARCH_LIMIT = 10
# Reranker is currently disabled in the chat path.
# RERANK_TOP_K = 5
MAX_CONTEXT_CHARS = 12000
MAX_REWRITE_HISTORY_MESSAGES = 6


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: str = None

class ChatResponse(BaseModel):
    query: str
    context: List[str]
    response: str
    session_id: str

# calling the prompts here from yaml file here we change the versions
with open("prompts/versions/v3.yaml", "r", encoding="utf-8") as f:
    prompt_config = yaml.safe_load(f)

SYSTEM_PROMPT = prompt_config["system_prompt"]
USER_PROMPT = prompt_config["user_prompt"]  



def build_context(retrieved_chunks: list[str]):
    if not retrieved_chunks:
        return "No relevant information found in the knowledge base."
    
    pdf_chunks = []
    metadata_chunks = []

    for chunk in retrieved_chunks:
        if any(marker in chunk for marker in ("CONTEXTUAL IMAGE", "MARKDOWN_EMBED", "IMAGE_URL")):
            metadata_chunks.append(chunk)
        else:
            pdf_chunks.append(chunk)

    context = ""
    if pdf_chunks:
        context += "=== MANUAL CONTENT ===\n"
        context += "\n\n".join(pdf_chunks)
    
    if metadata_chunks:
        if context:
            context += "\n\n"
        context += "=== IMAGE METADATA ===\n"
        context += "\n\n".join(metadata_chunks)

    return context if context else "No relevant information found in the knowledge base."


def fit_chunks_to_context_budget(chunks: list[str], max_chars: int = MAX_CONTEXT_CHARS) -> list[str]:
    selected_chunks = []
    used_chars = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        separator_chars = 2 if selected_chunks else 0
        remaining_chars = max_chars - used_chars - separator_chars

        if remaining_chars <= 0:
            break

        if len(chunk) <= remaining_chars:
            selected_chunks.append(chunk)
            used_chars += len(chunk) + separator_chars
        elif remaining_chars > 500:
            selected_chunks.append(chunk[:remaining_chars].rstrip())
            break

    return selected_chunks



   

# reranker code 
def rerank_chunks(query, chunks, top_k=10):
    """
    chunks: list of retrieved text chunks
    query: user question
    """

    if not chunks:
        return []

    # build pairs (query, chunk)
    pairs = [(query, chunk) for chunk in chunks]

    # get relevance scores
    scores = model_manager.reranker.predict(pairs)

    # attach scores
    scored_chunks = list(zip(chunks, scores))

    # sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # return top_k chunks
    top_chunks = [chunk for chunk, score in scored_chunks[:top_k]]

    return top_chunks

class QdrantHybridCustomRetriever(BaseRetriever):
    def __init__(self, collectionName:str, search_limit:int):
        self._collectionName = collectionName
        self._search_limit = search_limit

        super().__init__()

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str  

        # Generate embedding for dense search
        query_embedding = model_manager.embedding_model.encode(
            query_str,
            normalize_embeddings=True
        ).tolist()
        print("Embedding generated")

        #Generate embedding for sparse search
        query_sparse = list(
            model_manager.sparse_embedding.embed([query_str])
        )[0]

        print("type of query", type(query_embedding))
        # dense prefetch
        dense_prefetch = Prefetch(
            query = query_embedding,
            using ="dense"
        )


        # Sparse prefetch
        sparse_prefetch = Prefetch(
            query = SparseVector(
                   indices=query_sparse.indices,
                   values=query_sparse.values,
            ),
            using = SPARSE_VECTOR_NAME
        )


        # Search Qdrant
        search_result = qdrant_client.query_points(
            collection_name=self._collectionName,
            query =FusionQuery(
                fusion = Fusion.RRF
            ) ,
            prefetch = [
                dense_prefetch,
                sparse_prefetch
            ],
            limit=self._search_limit,
            with_payload=True
        )
        print(f"Found {len(search_result.points)} search results")

        nodes =[]
        seen_hash_text = set()

        for point in search_result.points:
            payload = point.payload or {}
            text = payload.get("text")

            if not text:
                continue

            hash_text = hashlib.md5(text.encode("utf-8")).hexdigest()
            if hash_text in seen_hash_text:
                continue

            seen_hash_text.add(hash_text)

            node = TextNode(
                text = text,
                metadata = {
                    "page": payload.get("page", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "doc_id": payload.get("doc_id", "")
                }
            ) 

            nodes.append(NodeWithScore(node = node, score=point.score or 0.0))
        return nodes    


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        start = time.perf_counter()
        print(f"Chat request received: {request.query}")


        session_id = request.session_id
        if not session_id:
            session_id = generate_session_id(None)
            if not session_id:
                session_id = str(uuid.uuid4())
        memory = get_chat_memory(session_id)


        # collecting info of vector db
        collect_info = qdrant_client.count(
            collection_name=COLLECTION_NAME,
            exact= True
        )

        print("collection info", collect_info)

        custom_retriver = QdrantHybridCustomRetriever(COLLECTION_NAME, SEARCH_LIMIT)

        retriver = custom_retriver.retrieve(request.query)

        retrieved_chunks = [
            node.node.text for node in retriver
        ]
        
        # Reranker disabled for now. Use Qdrant's hybrid RRF ordering directly.
        # reranked_chunks = rerank_chunks(request.query, retrieved_chunks, top_k=RERANK_TOP_K)
        # context_chunks = fit_chunks_to_context_budget(retrieved_chunks)

        # print("Context chunks after budget:", len(context_chunks))

     

        context_text = build_context(retrieved_chunks)

        print("==========CONTEXT SENT TO LLM========")
        print(f"Context Length: {len(context_text)}")
        # can also print context text for deebugging here

        

        # Creating CondensePlusContextChatEngine for having context of querys
        condense_chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever = custom_retriver,
            memory = memory,
            llm = google_llm,
            system_prompt = SYSTEM_PROMPT,
            context_prompt = USER_PROMPT.replace(
                "{context}", "{context_text}"
            ).replace(
                "{query}", "{query_str}"
            ),
            verbose = False
        )

        
        # message = [
        #     ChatMessage(
        #         role="system",
        #         content=SYSTEM_PROMPT 
        #     ),
            
        #     ChatMessage(
        #         role="user",
        #         content=USER_PROMPT.format(
        #             context=context_text, 
        #             query=request.query
        #         )
        #     )    
        # ]

        # print("Messages prepared for LLM")

        # Watsonx API Call
        try:
            # response = watsonx_llm.chat(
            #     message,
            #     temperature = 0.3
            # )

            response =await condense_chat_engine.achat(request.query)
            
            print("Response raw:", response)

                        
            # Extract text from response - handle different response formats
            llm_response_text = ""
            try:
                # 1. Primary Extraction: Direct LlamaIndex Message Attributes
                if hasattr(response, "message") and response.message:
                    if hasattr(response.message, "content") and response.message.content:
                        llm_response_text = str(response.message.content)
                    elif "content" in getattr(response.message, "additional_kwargs", {}):
                        llm_response_text = str(response.message.additional_kwargs["content"])
                
                # 2. Secondary Extraction: Top-level LlamaIndex Attributes
                if not llm_response_text or llm_response_text.strip() == "":
                    if hasattr(response, "text") and response.text:
                        llm_response_text = str(response.text)
                    elif hasattr(response, "response") and response.response:
                        llm_response_text = str(response.response)

                # 3. Third Extraction: Direct string casting safety fallback
                if not llm_response_text or llm_response_text.strip() == "":
                    llm_response_text = str(response)

            except Exception as extraction_err:
              print(f"Silent Extraction Error Handled: {str(extraction_err)}")
              llm_response_text = "Parsing error: Unable to unpack LLM text stream."
                
            if not llm_response_text or llm_response_text.strip() == "" or llm_response_text == "None":
                  llm_response_text = "The LLM completed execution successfully but emitted a null token stream due to prompt constraints."

            print(f"--- Sending Output Response to Client --- \n{llm_response_text[:100]}...")   
        except Exception as llm_error:
            print(f"LLM Error: {str(llm_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"LLM call failed: {str(llm_error)}"
            )
        
        end = time.perf_counter()

        print(f"==========Execution time in chat route: {end - start:.6f} seconds=========")
        return ChatResponse(
            query=request.query,
            context=retrieved_chunks,
            response=llm_response_text,
            session_id= session_id
        )
    


    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in chat endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

