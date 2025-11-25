import google.generativeai as genai
from PIL import Image

class ChartOCR:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def extract_insights(self, image_path):
        img = Image.open(image_path)

        prompt = """
You are a Chart OCR system. Extract:
1. Chart type
2. X and Y axis labels
3. All numerical values visible
4. Trend summary
5. Key business insights
Provide results in JSON format.
"""

        response = self.model.generate_content([prompt, img])
        return response.text
