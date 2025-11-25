import google.generativeai as genai
from PIL import Image

class ImageEmbedder:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def embed(self, image_path):
        image = Image.open(image_path)

        embedding = genai.embed_content(
            model="models/embedding-001",
            content=image
        )

        return embedding["embedding"]
