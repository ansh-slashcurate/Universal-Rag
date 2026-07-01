import os
import uuid
import magic
import hashlib
from fastapi import APIRouter, UploadFile, File, HTTPException
from parser.liteparser.liteparser import lite_parser, dockling_parser
from parser.pyMupdf.extract_metadataV2 import (
    extract_image_metadata,
    metadata_to_text,
)
from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    JSONNodeParser as LlamaJSONNodeParser,
)
from models.embedding.emb_model import model_manager
from utils import COLLECTION_NAME
from db.db_client import qdrant_client
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, SparseVector
from parser.jsonReader.jsonReader import json_reader
from utils import (
    ALLOWED_MIME_TYPES,
    JSON_MIME_TYPES,
    MIME_TYPE_BY_EXTENSION,
    SUPPORTED_UPLOAD_EXTENSIONS,
)
from retrieval.hybrid_search import SPARSE_VECTOR_NAME
import time
from parser.liteparser.render_pages import blocks_to_markdown  # render_pages, is_table_chunk commented out
from parser.liteparser.paddleStructure import PageBlock
# from parser.liteparser.processTablePage import process_table_page
import re


router = APIRouter()

UPLOAD_DIRECTORY = "uploads"


# Instantiate parsers once at module level
json_node_parser = LlamaJSONNodeParser()
hierarchical_node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 1024, 512])  

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def normalize_mime_type(mime_type: str, filename: str | None) -> str:
    filename = (filename or "").lower()
    extension = os.path.splitext(filename.lower())[1]

    if mime_type == "text/plain" and filename.endswith(".json"):
        return "application/json"
    if mime_type == "text/plain" and filename.endswith(".jsonl"):
        return "application/x-ndjson"
    if mime_type == "text/plain" and filename.endswith(".ndjson"):
        return "application/x-ndjson"

    extension_mime_type = MIME_TYPE_BY_EXTENSION.get(extension)
    if extension_mime_type and (
        mime_type
        in {
            "application/octet-stream",
            "application/zip",
            "application/x-zip-compressed",
            "text/plain",
            "text/xml",
        }
        or mime_type not in ALLOWED_MIME_TYPES + JSON_MIME_TYPES
    ):
        return extension_mime_type

    return mime_type


def ensure_payload_index(field_name: str):
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        message = str(e).lower()
        if "already exists" not in message:
            print(f"Payload index creation skipped for {field_name}: {str(e)}")


def build_payload_filter(field_name: str, value: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key=field_name,
                match=MatchValue(value=value),
            )
        ]
    )


def get_cached_file_info(file_hash: str):
    file_filter = build_payload_filter("file_hash", file_hash)

    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=file_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        chunk_count = qdrant_client.count(
            collection_name=COLLECTION_NAME,
            count_filter=file_filter,
            exact=True,
        ).count

        payload = points[0].payload or {}
        return {
            "chunks": chunk_count,
            "doc_id": payload.get("doc_id"),
        }
    except Exception as e:
        print(f"Cache lookup bypassed: {str(e)}")
        return None



@router.post("/upload")
async def upload_file(files: list[UploadFile] = File(...)):

    # starting time counter
    start = time.perf_counter()

    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    upload_results = []

    try:
        ensure_payload_index("doc_id")
        ensure_payload_index("file_hash")

        for file in files:
            # Localize document tracker arrays per file to prevent crossover contamination
            documents = []
            leaf_nodes = []
            image_metadata_entries = []
            parsed_pages = []
            page_images = {}

            safe_filename = os.path.basename(file.filename or f"{uuid.uuid4()}.pdf")
            display_filename = file.filename or safe_filename
            file_location = os.path.join(UPLOAD_DIRECTORY, safe_filename)

            content = await file.read()
            file_hash = get_file_hash(content)

            # Detect strict file signature types
            mime_type = magic.from_buffer(content, mime=True)
            print("mime type", mime_type)

            mime_type = normalize_mime_type(mime_type, file.filename)
                
            # Validation checkpoint
            if mime_type not in ALLOWED_MIME_TYPES and mime_type not in JSON_MIME_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type: {mime_type}. "
                        f"Allowed extensions: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
                    ),
                )

            # cached_file = get_cached_file_info(file_hash)
            # if cached_file:
            #     upload_results.append(
            #         {
            #             "filename": display_filename,
            #             "pages": 0,
            #             "image_metadata": 0,
            #             "chunks": cached_file["chunks"],
            #             "vectors": cached_file["chunks"],
            #             "status": "done",
            #             "cached": True,
            #             "file_hash": file_hash,
            #         }
            #     )
            #     print(f"Cache hit for {display_filename}; already indexed as {cached_file.get('doc_id')}")
            #     continue

            # Write only uncached files for parsers that require a local path.
            with open(file_location, "wb") as f:
                f.write(content)

            # --- ROUTE A: Standard Document Handling & Visual Layouts ---
            if mime_type in ALLOWED_MIME_TYPES:

                # calling docling parser for parsing
                parsed_result = dockling_parser(file_location)

                # Run metadata image extraction layers strictly on PDF instances
                # WARNING: Do not remove or touch the extract_metadata code
                if mime_type == "application/pdf":
                    image_metadata_entries = extract_image_metadata(file_location)
                    for image_metadata in image_metadata_entries:
                        page_number = image_metadata.get("page", 1)
                        page_images.setdefault(page_number, [])
                        page_images[page_number].append(metadata_to_text(image_metadata))

                # Build Document contexts page-by-page directly from Docling result
                num_pages = len(parsed_result.pages) if (hasattr(parsed_result, "pages") and parsed_result.pages) else 1
                for page_num in range(1, num_pages + 1):
                    try:
                        # docling's export_to_markdown has native layout and table formatting.
                        # image_mode="placeholder" keeps it lightweight and clean.
                        page_text = parsed_result.export_to_markdown(page_no=page_num, image_mode="placeholder")
                        print("="*100)
                        print("Parsed text", page_text)
                    except Exception as e:
                        print(f"Error exporting page {page_num} to markdown: {str(e)}")
                        page_text = ""

                    associated_images = page_images.get(page_num, [])
                    if associated_images:
                        page_text += "\n\n### ASSOCIATED DIAGRAM METADATA \n" + "\n\n".join(associated_images)

                    documents.append(
                        Document(
                            text=page_text,
                            metadata={
                                "filename": display_filename,
                                "page": page_num,
                                "source_type": "manual",
                                "related_image_count": len(associated_images),
                            },
                        )
                    )

                    # Append image insights as decoupled vector context strings
                    # for image_text in associated_images:
                    #     documents.append(
                    #         Document(
                    #             text=image_text,
                    #             metadata={
                    #                 "filename": display_filename,
                    #                 "page": page_num,
                    #                 "source_type": "image",
                    #             }
                    #         )
                    #     )

                # Execute structural parent-child markdown splitting
                if documents:

                    all_nodes = hierarchical_node_parser.get_nodes_from_documents(documents)
                    print("+"*100)
                    print("all node", all_nodes)
                    leaf_nodes = get_leaf_nodes(all_nodes)

                    print("+"*100)
                    print("leaf node", leaf_nodes)

            # --- ROUTE B: Structured Core JSON Processing ---
            elif mime_type in JSON_MIME_TYPES:
                print("Compiler is at else josn code")
                json_documents = json_reader.load_data(input_file = file_location)

                for doc in json_documents:
                    doc.metadata.update({
                            "filename": display_filename,
                            "page": 1,
                            "source_type": "structured_json"
                        })      
                # leaf_nodes = json_node_parser.get_nodes_from_documents(json_documents)
                leaf_nodes = json_documents
                print("Leaf Nodes", leaf_nodes)
                # Apply fallback metadata structural defaults
                for node in leaf_nodes:
                    node.metadata.setdefault("filename", display_filename)
                    node.metadata.setdefault("page", 1)
                    node.metadata.setdefault("source_type", "structured_json")

                
            # --- VECTOR PIPELINE LAYER ---
            embedding_records = []
            for chunk_index, node in enumerate(leaf_nodes):
                embedding_records.append(
                    {
                        "text": node.text,
                        "payload": {
                            "text": node.text,
                            "doc_id": display_filename,
                            "page": node.metadata.get("page", 1),
                            "chunk_index": chunk_index,
                            "source_type": node.metadata.get("source_type", "manual"),
                            "node_id": node.node_id,
                            "file_hash": file_hash,
                            "related_images": page_images.get(node.metadata.get("page", 1), []),
                        },
                    }
                )

            if not embedding_records:
                continue  # Cleanly move to the next file if no contents were parsed

            # Bulk Compute Tensor Embeddings locally
            texts = [record["text"] for record in embedding_records]

            vectors =model_manager.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size= 64
            )

            sparse_vectors = list(
                model_manager.sparse_embedding.embed(texts)
            )

            print("type of sparser vectors",type(sparse_vectors[0]))
            print("dir",dir(sparse_vectors[0]))

            print("====Sparsse vecrot")
            print("sparse", sparse_vectors[0])

                
            

            # Purge any existing references to this exact filename to prevent ghost document pollution
            try:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=build_payload_filter("doc_id", display_filename)
                )
            except Exception as e:
                print(f"Purge stage bypassed or historical reference missing: {str(e)}")

            # Batch up Point structures

            try:
                points = []
                for i, record in enumerate(embedding_records):
                    chunk_id = hashlib.md5(
                        f"{display_filename}_{record['payload']['page']}_{record['payload']['chunk_index']}".encode()
                    ).hexdigest()

                    points.append(
                        PointStruct(
                            id=chunk_id,
                            vector={
                                "dense": vectors[i].tolist(),
                                SPARSE_VECTOR_NAME: SparseVector(
                                indices=sparse_vectors[i].indices.tolist(),
                                values=sparse_vectors[i].values.tolist(),
                            ),
                            },
                            payload=record["payload"]
                         )
                    )

                # Atomic Upsert block execution
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
            except Exception as e:
                print("Error at Point Struct",e)    
            

            upload_results.append(
                {
                    "filename": display_filename,
                    "pages": len(parsed_result.pages) if parsed_result.pages else 1,
                    "image_metadata": len(image_metadata_entries),
                    "chunks": len(leaf_nodes),
                    "vectors": len(points),
                    "status": "done",
                    "cached": False,
                    "file_hash": file_hash,
                }
            )

            print("uploaded result", upload_results)

            end = time.perf_counter()

            print(f"Execution time in processing file{file.filename} is: {end - start:.6f} seconds")

    except HTTPException:
        raise
    except Exception as e:
        print("Exception ",e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index upload sequence: {str(e)}",
        )

    return {
        "files": upload_results,
        "message": "Done.",
    }
