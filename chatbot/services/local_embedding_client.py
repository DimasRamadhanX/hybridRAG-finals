import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Menginisialisasi model Hugging Face (Otomatis menghasilkan 384 dimensi)
try:
    hf_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("✅ Model Hugging Face all-MiniLM-L6-v2 berhasil dimuat lokal.")
except Exception as e:
    logger.error(f"❌ Gagal memuat model Hugging Face: {e}")
    hf_model = None

def get_local_embedding(text: str):
    """
    Mengubah teks menjadi embedding 384 dimensi secara OFFLINE
    menggunakan Hugging Face Sentence-Transformers.
    """
    if hf_model is None:
        raise RuntimeError("Model Hugging Face tidak tersedia di sistem lokal.")
        
    # generate embedding dan ubah numpy array menjadi list biasa
    embedding_vector = hf_model.encode(text)
    return embedding_vector.tolist()