from pathlib import Path
import fitz
import numpy as np
from PIL import Image 
import re
from parser.liteparser.paddleStructure import PageBlock

def render_pages(pdf_path: Path, dpi: int = 200) -> list[np.ndarray]:
    """Render every PDF page as an RGB numpy array."""
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    doc.close()
    return images



# checking if data has tables or not
def is_table_chunk(text: str, threshold: int = 3) -> bool:
    """
    True if this page's text chunk has enough number-dense lines
    to be considered a table page.
    """
    lines = text.strip().splitlines()
    number_lines = sum(
        1 for line in lines
        if sum(1 for t in line.split()
               if re.match(r"^\d+(\.\d+)?%?$", t)) >= 3
    )
    return number_lines >= threshold


def blocks_to_markdown(blocks: list[PageBlock]) -> str:
    """Stitch all page blocks into one markdown string."""
    parts = []
    for block in blocks:
        parts.append(f"<!-- page {block.page_num + 1} -->")
        parts.append(block.content.strip())
    return "\n\n".join(parts)