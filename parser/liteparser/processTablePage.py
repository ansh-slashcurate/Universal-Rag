from parser.liteparser.paddleStructure import _load, _engine, PageBlock
from parser.liteparser.htmlTableToMarkdown import html_table_to_markdown
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
        print("Error in load")
        return PageBlock(
            page_num=page_num,
            block_type="text",
            content=fallback_text,
        )
 
    try:

        output = _engine.pipeline.predict(input=page_img)
        print("="*50)
        print("Output respose keys")
        for res in output:
            print(res.keys() if hasattr(res, "keys") else type(res))
 
        
 
    except Exception as e:
        print("error in getting output from ppstructre", e)
        return PageBlock(
            page_num=page_num,
            block_type="text",
            content=fallback_text,
        )

    md_parts = []
    for res in output:
        # TableRecognitionPipelineV2 result includes table HTML per detected table
        tables = res.get("table_res_list", []) if hasattr(res, "get") else []
        for table in tables:
            html = table.get("pred_html", "")
            if html:
                md_parts.append(html_table_to_markdown(html))
 
    content = "\n\n".join(md_parts) if md_parts else fallback_text

 
    return PageBlock(
        page_num=page_num,
        block_type="table",
        content=content,
    )
 