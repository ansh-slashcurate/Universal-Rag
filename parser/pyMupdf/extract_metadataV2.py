"""
Image metadata extractor for RAG pipelines.

Key design decisions:
- Context window: pulls text from previous + current + next page so the
  step/caption that *describes* an image isn't missed when it lives on an
  adjacent page (very common in SOPs/manuals).
- Positional step matching: after gathering the context window, we find the
  step reference whose bbox is *closest above* the image on the page, not
  just the first step mention anywhere on the page.
- Dedup: MD5 hash dedup prevents logos/repeated headers from polluting the index.
- Structured metadata fields are kept flat and explicit so an LLM can parse
  each field independently without ambiguity.
"""

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


# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_IMAGES = "extracted_images"
OUTPUT_INDEX  = "image_metadata_index.txt"
OUTPUT_JSON   = "image_metadata.json"

# ── IBM COS config ────────────────────────────────────────────────────────────
COS_ENDPOINT    = os.getenv("COS_ENDPOINT")
COS_BUCKET      = os.getenv("COS_BUCKET")
COS_API_KEY_ID  = os.getenv("COS_API_KEY_ID")
COS_INSTANCE_CRN = os.getenv("COS_INSTANCE_CRN")
BASE_URL        = os.getenv("BASE_URL")
if not BASE_URL or BASE_URL.startswith(("f\"", "f'")):
    BASE_URL = (
        f"https://{COS_BUCKET}.s3.us-south.cloud-object-storage.appdomain.cloud"
        if COS_BUCKET else None
    )

# ── Tuning knobs ──────────────────────────────────────────────────────────────
BATCH_SIZE          = 20
MAX_WORKERS         = 5

MIN_IMAGE_WIDTH     = 250
MIN_IMAGE_HEIGHT    = 180
HEADER_MARGIN_PCT   = 0.08   # ignore images in top 8 % of page (headers)
FOOTER_MARGIN_PCT   = 0.92   # ignore images in bottom 8 % of page (footers)

# How many pages before/after the image page to include in the context window.
# Increase to 2 for very dense manuals where a figure caption may be two pages away.
CONTEXT_PAGES_BEFORE = 1
CONTEXT_PAGES_AFTER  = 1


# ── Text helpers ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and", "or",
    "with", "from", "this", "that", "it", "as", "be", "by", "are", "was",
    "were", "will", "can", "into", "then", "when", "please", "make", "sure",
    "after", "before", "below", "above", "select", "click", "put",
    "page", "menu", "press", "check", "also", "each", "next", "back",
    "open", "used", "only", "both", "have", "been", "during", "ensure",
    "need", "once", "note", "first", "while", "figure", "shown", "shows",
    "using", "refer", "image", "diagram", "illustration",
}


def sanitize(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def extract_keywords(text: str, n: int = 15) -> list[str]:
    """
    Extract the most meaningful technical words from text.
    Prefer longer words (more specific) and deduplicate.
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    seen, keywords = set(), []
    for word in words:
        if word not in STOPWORDS and word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) == n:
            break
    return keywords or ["equipment", "procedure", "sop"]


def find_best_caption(page_text: str, doc_name: str, page_num: int) -> str:
    """
    Pick the best single-line caption from page text.
    Priority: figure/diagram labels → first substantive sentence → fallback.
    """
    # 1. Explicit figure/diagram label (e.g. "Figure 3 – Valve assembly")
    fig_match = re.search(
        r"(?:Figure|Fig\.?|Diagram|Illustration|Image)\s*[\d\.]*\s*[:\-–]?\s*(.{10,80})",
        page_text, re.IGNORECASE
    )
    if fig_match:
        return fig_match.group(1).strip()

    # 2. First meaningful line (>20 chars, not a page number, not a header noise word)
    lines = [
        ln.strip() for ln in page_text.splitlines()
        if len(ln.strip()) > 20
        and not ln.strip().isdigit()
        and not re.match(r"^(page|chapter|\d+\.?\d*)\b", ln.strip(), re.IGNORECASE)
    ]
    if lines:
        return lines[0][:80].strip()

    return f"{doc_name} — Page {page_num}"


def find_step_for_image(page, img_bbox, page_text: str) -> str:
    """
    Find the step that is most relevant to a specific image on a page.

    Strategy:
    - Extract all step mentions with their vertical position on the page using
      PyMuPDF's search_for(), which returns bounding boxes.
    - Return the step whose text bbox is *closest above* the image bbox.
      This mirrors how humans read: the step label that precedes the figure
      is the one the figure illustrates.
    - Fall back to the last step mentioned anywhere on the page, then to
      a keyword search in the surrounding text.
    """
    step_pattern = re.compile(r"[Ss]tep\s*\d+[\.\:]?", re.IGNORECASE)

    # Gather all on-page step hits with their Y position
    step_hits = []  # list of (y_top, step_label)
    for match in step_pattern.finditer(page_text):
        label = match.group(0).strip()
        hits = page.search_for(label)
        for rect in hits:
            step_hits.append((rect.y0, label))

    if not step_hits:
        # Try broader pattern: "3.", "3.1", "Step Three" etc.
        broader = re.findall(r"\b(\d+\.\d*|\bStep\s+\w+)\b", page_text, re.IGNORECASE)
        if broader:
            return broader[0]
        return "General"

    img_top = img_bbox.y0

    # Steps that appear ABOVE the image
    above = [(y, label) for y, label in step_hits if y <= img_top]
    if above:
        # Closest above = highest Y value that is still <= img_top
        return max(above, key=lambda x: x[0])[1]

    # No step above → take the first step below (image is an intro illustration)
    return min(step_hits, key=lambda x: x[0])[1]


def build_context_window(pdf: fitz.Document, page_num: int) -> str:
    """
    Concatenate text from adjacent pages to form a richer context window.
    page_num is 0-indexed internally.
    """
    total = len(pdf)
    start = max(0, page_num - CONTEXT_PAGES_BEFORE)
    end   = min(total - 1, page_num + CONTEXT_PAGES_AFTER)
    parts = []
    for p in range(start, end + 1):
        text = pdf[p].get_text().strip()
        if text:
            parts.append(f"[Page {p + 1}]\n{text}")
    return "\n\n".join(parts)


def make_description(caption: str, doc_name: str, page_num: int, step: str, keywords: list[str]) -> str:
    kw_str = ", ".join(keywords[:6])
    return (
        f"{step} visual from '{doc_name}', page {page_num}. "
        f"{caption}. "
        f"Key topics: {kw_str}."
    )


def metadata_to_text(entry: dict) -> str:
    """
    Flat, labelled format optimised for LLM retrieval.
    Each field on its own line with a fixed-width label so the model can
    reliably parse field boundaries.
    """
    return "\n".join([
        f"DOCUMENT       : {entry['document']}",
        f"PAGE           : {entry['page']}",
        f"STEP           : {entry['step']}",
        f"CAPTION        : {entry['caption']}",
        f"DESCRIPTION    : {entry['description']}",
        f"IMAGE_URL      : {entry['url']}",
        f"MARKDOWN_EMBED : {entry['render']}",
        f"KEYWORDS       : {', '.join(entry['keywords'])}",
        f"INDEXED_AT     : {entry['indexed_at']}",
    ])


# ── COS helpers ───────────────────────────────────────────────────────────────

def init_cos():
    print("Connecting to IBM COS...")
    missing = [
        name for name, val in {
            "COS_ENDPOINT": COS_ENDPOINT,
            "COS_BUCKET": COS_BUCKET,
            "COS_API_KEY_ID": COS_API_KEY_ID,
            "COS_INSTANCE_CRN": COS_INSTANCE_CRN,
            "BASE_URL": BASE_URL,
        }.items() if not val
    ]
    if missing:
        raise ValueError(f"Missing COS env vars: {', '.join(missing)}")

    client = ibm_boto3.client(
        "s3",
        ibm_api_key_id=COS_API_KEY_ID,
        ibm_service_instance_id=COS_INSTANCE_CRN,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT,
    )
    client.list_buckets()
    print("IBM COS connected OK\n")
    return client


def upload_bytes(cos, key: str, data, content_type: str = "image/png") -> str | None:
    try:
        body = data.encode("utf-8") if isinstance(data, str) else data
        cos.put_object(
            Bucket=COS_BUCKET,
            Key=key,
            Body=body,
            ContentType=content_type,
            ACL="public-read",
        )
        return f"{BASE_URL}/{key}"
    except Exception as exc:
        print(f"  UPLOAD FAILED [{key}] — {type(exc).__name__}: {exc}")
        return None


# ── Per-image processing ──────────────────────────────────────────────────────

def process_image(task: dict, cos) -> dict | None:
    """
    Upload image + metadata to COS and return the metadata entry dict.
    All heavy text analysis happens here so it can run in parallel workers.
    """
    try:
        # Write local copy
        os.makedirs(os.path.dirname(task["filepath"]), exist_ok=True)
        with open(task["filepath"], "wb") as f:
            f.write(task["img_bytes"])

        img_url = upload_bytes(cos, f"images/{task['filename']}", task["img_bytes"])
        if not img_url:
            return None

        # ── Core metadata extraction ──────────────────────────────────────────
        # Use the multi-page context window for keywords/description so we don't
        # miss context that lives on the page before or after the image.
        context_text = task["context_text"]
        page_text     = task["page_text"]     # single-page text for step matching

        step     = task["step"]               # already resolved positionally in collect()
        keywords = extract_keywords(context_text)
        caption  = find_best_caption(context_text, task["doc_name"], task["page_num"])
        description = make_description(caption, task["doc_name"], task["page_num"], step, keywords)

        entry = {
            "image_id"   : task["filename"],
            "document"   : task["doc_name"],
            "page"       : task["page_num"],
            "step"       : step,
            "caption"    : caption,
            "description": description,
            "keywords"   : keywords,
            "url"        : img_url,
            "render"     : f"![{caption}]({img_url})",
            "indexed_at" : datetime.now(timezone.utc).isoformat(),
        }

        # Upload text metadata alongside the image
        meta_key = f"metadata/{task['filename'].replace('.png', '')}_meta.txt"
        upload_bytes(cos, meta_key, metadata_to_text(entry), "text/plain")

        return entry

    except Exception as exc:
        print(f"  ERROR [{task['filename']}] — {type(exc).__name__}: {exc}")
        return None


# ── PDF scanning ──────────────────────────────────────────────────────────────

def collect_image_tasks(pdf_path: str, output_images: str = OUTPUT_IMAGES) -> list[dict]:
    """
    Scan the PDF and build the task list for parallel processing.

    The critical fix vs the original:
    - img_bbox is obtained BEFORE closing the page so we can call search_for()
      on it to resolve the step positionally.
    - context_text (multi-page window) is built here and stored in the task so
      workers don't need to re-open the PDF.
    - Logo/repeated-image dedup via MD5 is global across the whole document.
    """
    all_tasks = []
    global_hash_counts: dict[str, int] = {}

    pdf_file = os.path.basename(pdf_path)
    doc_name = sanitize(pdf_file)
    pdf = fitz.open(pdf_path)

    try:
        img_count = 0
        for page_idx in range(len(pdf)):
            page      = pdf[page_idx]
            page_text = page.get_text()
            page_h    = page.rect.height

            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref     = img[0]
                    base_img = pdf.extract_image(xref)
                    img_bytes = base_img["image"]
                    width, height = base_img["width"], base_img["height"]

                    # ── Size filter ───────────────────────────────────────────
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue

                    # ── Position filter (skip header/footer band) ─────────────
                    img_bbox = page.get_image_bbox(img)
                    if (img_bbox.y0 < page_h * HEADER_MARGIN_PCT or
                            img_bbox.y1 > page_h * FOOTER_MARGIN_PCT):
                        continue

                    # ── Dedup repeated images (logos, watermarks) ─────────────
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    global_hash_counts[img_hash] = global_hash_counts.get(img_hash, 0) + 1
                    if global_hash_counts[img_hash] > 1:
                        continue

                    # ── Positional step resolution (the main fix) ─────────────
                    # Done here while the page object is still open so we can
                    # call page.search_for() for bbox-aware matching.
                    step = find_step_for_image(page, img_bbox, page_text)

                    # ── Multi-page context window ─────────────────────────────
                    context_text = build_context_window(pdf, page_idx)

                    filename = f"{doc_name}_p{page_idx + 1}_img{img_idx + 1}.png"
                    filepath = os.path.join(output_images, filename)

                    all_tasks.append({
                        "doc_name"    : doc_name,
                        "pdf_file"    : pdf_file,
                        "img_bytes"   : img_bytes,
                        "page_text"   : page_text,
                        "context_text": context_text,
                        "page_num"    : page_idx + 1,
                        "img_bbox"    : img_bbox,   # kept for debugging
                        "step"        : step,
                        "filename"    : filename,
                        "filepath"    : filepath,
                    })
                    img_count += 1

                except Exception:
                    continue

        print(f"  {pdf_file} → {img_count} diagrams queued for processing.")
    finally:
        pdf.close()

    return all_tasks


# ── Index writing ─────────────────────────────────────────────────────────────

def write_index_files(entries: list[dict], cos,
                      output_index: str = OUTPUT_INDEX,
                      output_json: str = OUTPUT_JSON) -> None:
    print("\nWriting index files…")

    with open(output_index, "w", encoding="utf-8") as f:
        f.write("IMAGE REFERENCE INDEX\n")
        f.write("=" * 60 + "\n\n")
        for entry in entries:
            f.write(metadata_to_text(entry))
            f.write("\n" + "-" * 60 + "\n\n")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    with open(output_index, "rb") as f:
        upload_bytes(cos, "index/image_metadata_index.txt", f.read(), "text/plain")
    with open(output_json, "rb") as f:
        upload_bytes(cos, "index/image_metadata.json", f.read(), "application/json")

    print(f"  Index uploaded ({len(entries)} entries).")


# ── Public entry point ────────────────────────────────────────────────────────

def extract_image_metadata(pdf_path: str, output_images: str = OUTPUT_IMAGES) -> list[dict]:
    """
    Extract diagram images from a PDF, upload to IBM COS, and return
    a list of metadata dicts ready for embedding into a vector store.
    """
    start = time.time()
    os.makedirs(output_images, exist_ok=True)

    cos = init_cos()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Scanning PDF for diagrams…")
    tasks = collect_image_tasks(pdf_path, output_images=output_images)

    if not tasks:
        print("No images matched the filter criteria.")
        return []

    print(f"\nProcessing {len(tasks)} images in batches of {BATCH_SIZE}…\n")
    entries: list[dict] = []
    current_pdf = None

    for batch_start in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[batch_start : batch_start + BATCH_SIZE]

        for task in batch:
            if task["pdf_file"] != current_pdf:
                current_pdf = task["pdf_file"]
                print(f"--- {current_pdf} ---")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_image, task, cos): task for task in batch}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    entries.append(result)
                    print(
                        f"  ✓  p{result['page']:>3} | {result['step']:<14} | {result['caption'][:50]}"
                    )

    write_index_files(entries, cos)
    print(f"\nDone in {time.time() - start:.2f}s — {len(entries)} images indexed.")
    return entries