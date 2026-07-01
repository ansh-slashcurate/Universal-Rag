from liteparse import LiteParse
from docling.document_converter import DocumentConverter

parser = LiteParse(
    ocr_enabled=True,              # Enable OCR (default: True)
    ocr_language="eng",            # Tesseract language code
    # ocr_server_url= "http://localhost:8829/ocr",           # HTTP OCR server URL (optional)
    tessdata_path=None,            # Path to tessdata directory (optional)
    max_pages=1000,                # Max pages to parse
    dpi=150,                       # Rendering DPI
    preserve_very_small_text=False, # Keep tiny text
    password=None,                 # Password for protected documents
    quiet=False,                   # Suppress progress output
    num_workers=4,                 # Concurrent OCR workers
    output_format = "markdown",
    image_mode ="placeholder",
    extract_links = True
    
)

converter = DocumentConverter()

def lite_parser(file: str):
    """
    This function takes a file as input and returns the parsed content using LiteParse.
    """
    try:
        parsed_content = parser.parse(file)
        return parsed_content
    except Exception as e:
        raise Exception(f"Failed to parse file: {str(e)}")  

def dockling_parser(file:str):
    try:
        document = converter.convert(file)
        return document.document
    except Exception as e:
        raise Exception(f"Failed to parse file using docling: {str(e)}") 


    