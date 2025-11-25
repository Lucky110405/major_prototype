import pdfplumber
import tabula
import google.generativeai as genai
from PIL import Image
import pandas as pd

class TableExtractor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel("gemini-1.5-pro")

    def extract_from_pdf(self, pdf_path):
        try:
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            if tables:
                return [t.to_dict() for t in tables]
        except:
            pass

        # fallback – OCR using LLM
        with pdfplumber.open(pdf_path) as pdf:
            results = []
            for page in pdf.pages:
                img = page.to_image(resolution=300).original
                results.append(self.extract_from_image(img))
            return results

    def extract_from_image(self, img):
        prompt = """
Extract all tables from this image.
Return clean JSON: rows, columns, values.
"""
        response = self.llm.generate_content([prompt, img])
        return response.text
