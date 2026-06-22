# chatbot/services/gemini_client.py
import os
from google import genai
from google.genai import types  # <-- Tambahkan ini untuk konfigurasi dimensi
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi client tanpa memaksa versi v1/v1beta (biarkan SDK mengaturnya otomatis)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str):
    """
    Mengubah string teks menjadi array embedding 768 dimensi 
    menggunakan model gemini-embedding-2 terbaru.
    """
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768) # <-- Mengunci output di 768 dimensi
    )
    return response.embeddings[0].values

def generate_answer(prompt: str) -> str:
    """
    Mengirimkan prompt RAG lengkap ke model gemini-2.5-flash
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text