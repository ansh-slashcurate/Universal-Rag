from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os


os.environ["PADDLEX_OFFLINE_MODE"] = "1"

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
    pipeline: object = None
    lang: str = "en"
 
_engine = _PaddleEngine()
 
 
def init_engines(lang: str = "en") -> None:
    """
    Pre-load PaddleStructure at startup so first parse isn't slow.
    Optional — engine is lazy-loaded on first use if not called.
    """
    _load(lang)
 
 
def _load(lang: str = "en") -> bool:
    if _engine.pipeline is not None:
        return True
    try:
        os.environ["PADDLEX_OFFLINE_MODE"] = "1"
        os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
        os.environ["GLOG_v"] = "0"


        from paddleocr import TableRecognitionPipelineV2
        _engine.pipeline = TableRecognitionPipelineV2(
                enable_mkldnn=False,
    
                # ── lightweight model variants — mobile not server ──
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
    
                # ── skip doc preprocessing — flatbed scans don't need it ──
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
        )
        _engine.lang = lang
        return True
    except Exception as e:
        print("error in load function", e)
        return False
 
 