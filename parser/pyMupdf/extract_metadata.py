import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  

import fitz
import ibm_boto3
from ibm_botocore.client import Config


OUTPUT_IMAGES = "extracted_images"
OUTPUT_INDEX = "image_metadata_index.txt"
OUTPUT_JSON = "image_metadata.json"

COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_BUCKET = os.getenv("COS_BUCKET")
COS_API_KEY_ID = os.getenv("COS_API_KEY_ID")
COS_INSTANCE_CRN = os.getenv("COS_INSTANCE_CRN")
BASE_URL = os.getenv("BASE_URL")
if not BASE_URL or BASE_URL.startswith("f\"") or BASE_URL.startswith("f'"):
    BASE_URL = f"https://{COS_BUCKET}.s3.us-south.cloud-object-storage.appdomain.cloud" if COS_BUCKET else None


BATCH_SIZE = 20
MAX_WORKERS = 5

MIN_IMAGE_WIDTH = 250
MIN_IMAGE_HEIGHT = 180
HEADER_MARGIN_PCT = 0.08
FOOTER_MARGIN_PCT = 0.92


def sanitize(filename):
    """Remove spaces and special chars from filename."""
    name = os.path.splitext(filename)[0]
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def extract_keywords(page_text):
    """Pull meaningful technical keywords from surrounding page text."""
    stopwords = {
        "the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and", "or",
        "with", "from", "this", "that", "it", "as", "be", "by", "are", "was",
        "were", "will", "can", "into", "then", "when", "please", "make", "sure",
        "after", "before", "below", "above", "select", "click", "put",
        "page", "menu", "press", "check", "also",
        "each", "next", "back", "open", "used", "only", "both", "have", "been",
        "during", "ensure", "need", "once", "note", "first", "then", "while"
    }
    words = re.findall(r"\b[a-zA-Z]{4,}\b", page_text.lower())
    seen, keywords = set(), []
    for word in words:
        if word not in stopwords and word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) == 15:
            break
    return keywords if keywords else ["sop", "equipment", "procedure"]


def find_step(page_text):
    """Detect step reference from page text."""
    matches = re.findall(r"[Ss]tep\s*\d+", page_text)
    return matches[0] if matches else "General"


def make_caption(page_text, doc_name, page_num):
    """Use first meaningful line of page text as image caption."""
    lines = [
        line.strip() for line in page_text.splitlines()
        if len(line.strip()) > 15 and not line.strip().isdigit() and "erba" not in line.lower()
    ]
    if lines:
        return lines[0][:60].strip()
    return f"{doc_name} Page {page_num}"


def make_description(page_text, doc_name, page_num, step):
    caption = make_caption(page_text, doc_name, page_num)
    return f"{step} image from {doc_name} page {page_num}. {caption}"


def metadata_to_text(entry):
    """Format image metadata for vector embedding and Watsonx parsing."""
    return "\n".join([
        f"DOCUMENT IDENTIFIER: {entry['document']}",
        f"MANUAL LOCATION    : Page {entry['page']}, {entry['step']}",
        f"TECHNICAL CAPTION  : {entry['caption']}",
        f"CONTEXTUAL IMAGE   : {entry['render']}",
        f"DETAILED CONTEXT   : {entry['description']}",
        f"SEARCH KEYWORDS    : {', '.join(entry['keywords'])}",
        f"INDEXED TIMESTAMP  : {entry['indexed_at']}"
    ])


def init_cos():
    print("Connecting to IBM COS...")
    try:
        missing_config = [
            name for name, value in {
                "COS_ENDPOINT": COS_ENDPOINT,
                "COS_BUCKET": COS_BUCKET,
                "COS_API_KEY_ID": COS_API_KEY_ID,
                "COS_INSTANCE_CRN": COS_INSTANCE_CRN,
                "BASE_URL": BASE_URL
            }.items()
            if not value
        ]
        if missing_config:
            raise ValueError(f"Missing COS environment variables: {', '.join(missing_config)}")

        client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=COS_API_KEY_ID,
            ibm_service_instance_id=COS_INSTANCE_CRN,
            config=Config(signature_version="oauth"),
            endpoint_url=COS_ENDPOINT
        )
        client.list_buckets()
        print("IBM COS connected OK\n")
        return client
    except Exception as exc:
        print(f"ERROR: COS connection failed - {exc}")
        return None


def upload_bytes(cos, key, data, content_type="image/png"):
    """Upload raw bytes to COS."""
    try:
        body = data.encode("utf-8") if isinstance(data, str) else data
        cos.put_object(
            Bucket=COS_BUCKET,
            Key=key,
            Body=body,
            ContentType=content_type,
            ACL="public-read"
        )
        return f"{BASE_URL}/{key}"
    except Exception as exc:
        print(f"  UPLOAD FAILED [{key}] - {type(exc).__name__}: {exc}")
        return None


def process_image(task, cos):
    """Build metadata metrics and push image/metadata to COS."""
    doc_name = task["doc_name"]
    img_bytes = task["img_bytes"]
    page_text = task["page_text"]
    page_num = task["page_num"]
    filename = task["filename"]
    filepath = task["filepath"]

    try:
        with open(filepath, "wb") as image_file:
            image_file.write(img_bytes)

        img_url = upload_bytes(cos, f"images/{filename}", img_bytes)
        if not img_url:
            return None

        step = find_step(page_text)
        keywords = extract_keywords(page_text)
        caption = make_caption(page_text, doc_name, page_num)
        description = make_description(page_text, doc_name, page_num, step)

        entry = {
            "image_id": filename,
            "document": doc_name,
            "page": page_num,
            "step": step,
            "description": description,
            "keywords": keywords,
            "caption": caption,
            "url": img_url,
            "render": f"![{caption}]({img_url})",
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }

        meta_key = f"metadata/{filename.replace('.png', '')}_meta.txt"
        upload_bytes(cos, meta_key, metadata_to_text(entry), "text/plain")

        return entry

    except Exception as exc:
        print(f"  ERROR [{filename}] - {type(exc).__name__}: {exc}")
        return None


def write_index_files(entries, cos, output_index=OUTPUT_INDEX, output_json=OUTPUT_JSON):
    print("\nWriting unified index files...")
    with open(output_index, "w", encoding="utf-8") as index_file:
        index_file.write("IMAGE REFERENCE INDEX - Medical Equipment SOPs\n")
        index_file.write("=" * 60 + "\n\n")
        for entry in entries:
            index_file.write("ENTRY\n")
            index_file.write(f"DOCUMENT IDENTIFIER: {entry['document']}\n")
            index_file.write(f"MANUAL LOCATION    : Page {entry['page']}, {entry['step']}\n")
            index_file.write(f"TECHNICAL CAPTION  : {entry['caption']}\n")
            index_file.write(f"CONTEXTUAL IMAGE   : {entry['render']}\n")
            index_file.write(f"DETAILED CONTEXT   : {entry['description']}\n")
            index_file.write(f"SEARCH KEYWORDS    : {', '.join(entry['keywords'])}\n")
            index_file.write("-" * 60 + "\n\n")

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(entries, json_file, indent=2, ensure_ascii=False)

    with open(output_index, "rb") as index_file:
        upload_bytes(cos, "index/image_metadata_index.txt", index_file.read(), "text/plain")
    with open(output_json, "rb") as json_file:
        upload_bytes(cos, "index/image_metadata.json", json_file.read(), "application/json")


def collect_image_tasks(pdf_path, output_images=OUTPUT_IMAGES):
    all_tasks = []
    global_logo_registry = {}
    pdf_file = os.path.basename(pdf_path)
    doc_name = sanitize(pdf_file)

    pdf = fitz.open(pdf_path)
    try:
        img_count = 0
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            images = page.get_images(full=True)
            page_text = page.get_text()
            page_height = page.rect.height

            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    base_img = pdf.extract_image(xref)
                    img_bytes = base_img["image"]
                    width = base_img["width"]
                    height = base_img["height"]

                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue

                    bbox = page.get_image_bbox(img)
                    if bbox.y0 < (page_height * HEADER_MARGIN_PCT) or bbox.y1 > (page_height * FOOTER_MARGIN_PCT):
                        continue

                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    global_logo_registry[img_hash] = global_logo_registry.get(img_hash, 0) + 1
                    if global_logo_registry[img_hash] > 1:
                        continue

                    filename = f"{doc_name}_p{page_num + 1}_img{img_idx + 1}.png"
                    filepath = os.path.join(output_images, filename)

                    all_tasks.append({
                        "doc_name": doc_name,
                        "img_bytes": img_bytes,
                        "page_text": page_text,
                        "page_num": page_num + 1,
                        "filename": filename,
                        "filepath": filepath,
                        "pdf_file": pdf_file
                    })
                    img_count += 1
                except Exception:
                    continue

        print(f"  {pdf_file} - Completed. Extracting {img_count} verified diagrams.")
    finally:
        pdf.close()

    return all_tasks


def extract_image_metadata(pdf_path, output_images=OUTPUT_IMAGES):
    """
    Extract diagram images from one uploaded PDF, upload images and metadata to COS,
    write local index files, and return metadata entries for embedding.
    """
    start_time = time.time()
    os.makedirs(output_images, exist_ok=True)

    cos = init_cos()
    if not cos:
        raise RuntimeError("COS connection failed")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found - {pdf_path}")

    print("Pre-scanning PDF and extracting target diagrams...")
    all_tasks = collect_image_tasks(pdf_path, output_images=output_images)

    total = len(all_tasks)
    if total == 0:
        print("\nNo items matched your target layout criteria.")
        return []

    print(f"\nProcessing in parallel batches of {BATCH_SIZE}...\n")
    all_entries = []
    current_pdf = None

    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_tasks[batch_start:batch_start + BATCH_SIZE]
        for task in batch:
            if task["pdf_file"] != current_pdf:
                current_pdf = task["pdf_file"]
                print(f"--- Now processing: {current_pdf} ---")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_image, task, cos): task for task in batch}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_entries.append(result)
                    print(f"  OK  p{result['page']:>3} | {result['step']:<12} | {result['caption'][:45]}")

    write_index_files(all_entries, cos)
    print(f"\nExecution complete in {time.time() - start_time:.2f}s. Maintenance logs created cleanly.")
    return all_entries
