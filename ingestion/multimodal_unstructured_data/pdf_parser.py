import pdfplumber
from PyPDF2 import PdfReader
import os
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
import re

def create_chunks(pages):
    chunks = []
    buffer = ""

    for p in pages:
        text = p["text"]

        # Split by headings using regex
        parts = re.split(r'\n(?=[A-Z][A-Za-z0-9 ,.-]{3,})', text)

        for part in parts:
            if len(part.strip()) > 50:
                chunks.append({
                    "page": p["page_num"],
                    "text": part.strip()
                })

    return chunks


def extract_images(path):
    if not FITZ_AVAILABLE:
        return []
    doc = fitz.open(path)
    images = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "page": page_number + 1,
                "image_bytes": base_image["image"],
                "format": base_image["ext"]
            })
    return images


def extract_tables(path):
    if not CAMELOT_AVAILABLE:
        return []
    try:
        tables = camelot.read_pdf(path, flavor="lattice", pages="all")
        result = []

        for t in tables:
            result.append(t.df.to_dict(orient="records"))

        return result

    except:
        return []


def extract_metadata(path):
    reader = PdfReader(path)
    meta = reader.metadata

    return {
        "title": meta.title if meta else "",
        "author": meta.author if meta else "",
        "producer": meta.producer if meta else "",
        "num_pages": len(reader.pages),
        "file_size_kb": round(os.path.getsize(path)/1024, 2)
    }


def extract_text(path):
    all_pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_pages.append({
                    "page_num": page.page_number,
                    "text": text.strip()
                })
    return all_pages


def parse_pdf(file_path):
    output = {}

    # Step 1: Metadata
    output["metadata"] = extract_metadata(file_path)

    # Step 2: Raw Text
    pages_text = extract_text(file_path)
    output["raw_text"] = pages_text

    # Step 3: Tables
    output["tables"] = extract_tables(file_path)

    # Step 4: Images
    output["images"] = extract_images(file_path)

    # Step 5: Semantic Chunks for RAG
    output["chunks"] = create_chunks(pages_text)

    return output
