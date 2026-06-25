from parser.liteparser.htmlTableToMarkdown import html_table_to_markdown
from parser.liteparser.paddleStructure import _load, _engine, PageBlock
import numpy as np

def process_table_page(
    page_num: int,
    page_img: np.ndarray,
    fallback_text: str = "",
    lang: str = "en",
) -> PageBlock:
    """
    Run PaddleStructure on a table page and return a PageBlock
    with markdown content. Falls back to Tesseract text from
    LiteParse if PaddleStructure fails or isn't installed.
 
    Args:
        page_num:       0-based page index
        page_img:       numpy RGB array of the page (from PyMuPDF)
        fallback_text:  Tesseract text from LiteParse
        lang:           language code for PaddleStructure (default "en")
 
    Returns:
        PageBlock with block_type="table", content as markdown
    """
    if not _load(lang):
        return PageBlock(
            page_num=page_num,
            block_type="text",
            content=fallback_text,
        )
 
    try:
        result = _engine.model(page_img)
    except Exception as e:
        return PageBlock(
            page_num=page_num,
            block_type="text",
            content=fallback_text,
        )
 
    md_parts: list[str] = []
 
    for region in result:
        region_type = region.get("type", "")
 
        if region_type == "table":
            html = region.get("res", {}).get("html", "")
            md_parts.append(html_table_to_markdown(html) if html else fallback_text)
 
        elif region_type in ("text", "title"):
            text = region.get("res", [])
            if isinstance(text, list):
                # PaddleOCR format: [[bbox, (text, confidence)], ...]
                lines = [item[1][0] for item in text if item and len(item) > 1]
                md_parts.append("\n".join(lines))
            elif isinstance(text, str):
                md_parts.append(text)
 
        elif region_type == "figure":
            md_parts.append("_[Figure]_")
 
    content = "\n\n".join(md_parts) if md_parts else fallback_text
 
    return PageBlock(
        page_num=page_num,
        block_type="table",
        content=content,
    )
 