from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class PageBlock:
    page_num: int
    block_type: str          # "text" | "table"
    content: str
    raw_html: Optional[str] = None
 
 
# ─────────────────────────────────────────────
#  Engine singleton (private)
# ─────────────────────────────────────────────
 
class _PaddleEngine:
    model: object = None
    lang: str = "en"
 
_engine = _PaddleEngine()
 
 
def init_engines(lang: str = "en") -> None:
    """
    Pre-load PaddleStructure at startup so first parse isn't slow.
    Optional — engine is lazy-loaded on first use if not called.
    """
    _load(lang)
 
 
def _load(lang: str = "en") -> bool:
    if _engine.model is not None:
        return True
    try:
        from paddleocr import PPStructure
        _engine.model = PPStructure(table=True, ocr=True, lang=lang, show_log=False)
        _engine.lang = lang
        return True
    except ImportError:
        return False
 
 