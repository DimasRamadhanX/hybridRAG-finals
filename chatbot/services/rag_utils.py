import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Tuple, Optional
from chatbot.services.gemini_client import client

logger = logging.getLogger(__name__)

# --- Key aliases yang mungkin muncul dari berbagai query Cypher ---
_FIELD_ALIASES: Dict[str, List[str]] = {
    "title":       ["title", "filmAsal", "filmRekomendasi", "judul", "nama", "f.title", "Film"],
    "score":       ["score", "rating", "ratingRekomendasi", "f.score", "skor", "Score"],
    "director":    ["director", "sutradara", "directedBy", "d.name", "Director"],
    "genres":      ["genres", "genre", "genreRekomendasi", "g.name", "listGenre", "Genres"],
    "description": ["description", "sinopsis", "deskripsi", "f.description", "Description"],
}

def _extract_field(item: Dict[str, Any], field: str) -> Any:
    """Cari nilai record berdasarkan daftar alias field."""
    for alias in _FIELD_ALIASES.get(field, [field]):
        if alias in item and item[alias] is not None:
            return item[alias]
    return None

def normalize_graph_context(raw_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalisasi semua record mentah Neo4j ke struktur seragam."""
    normalized = []
    for item in raw_context:
        score_raw = _extract_field(item, "score")
        
        if isinstance(score_raw, (int, float)):
            # JARING PENGAMAN: Jika angka di bawah 5.0 (pasti skor PageRank GDS), 
            # kalikan 100 agar menjadi persentase bobot pengaruh yang indah (misal 1.25 -> 125%)
            if score_raw < 5.0:
                score_display = round(score_raw * 100)
            else:
                score_display = round(score_raw)
        else:
            score_display = None
        genres_raw = _extract_field(item, "genres")
        if isinstance(genres_raw, list):
            genres_clean = [g for g in genres_raw if g]
        elif isinstance(genres_raw, str) and genres_raw:
            genres_clean = [genres_raw]
        else:
            genres_clean = []

        normalized.append({
            "title":       _extract_field(item, "title") or "Film Tanpa Judul",
            "score":       score_display,
            "director":    _extract_field(item, "director"),
            "genres":      genres_clean,
            "description": _extract_field(item, "description"),
            "_raw": {
                k: v for k, v in item.items()
                if k not in {a for aliases in _FIELD_ALIASES.values() for a in aliases}
            },
        })
    return normalized

def call_gemini_with_retry(
    prompt: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Panggil Gemini dengan exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text, None
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                error_type = "quota"
            elif "401" in err_str or "403" in err_str or "api_key" in err_str:
                error_type = "auth"
            elif "timeout" in err_str or "connection" in err_str:
                error_type = "network"
            else:
                error_type = "unknown"

            logger.warning(f"[Gemini attempt {attempt}/{max_retries}] {error_type}: {e}")

            # 🌟 PERBAIKAN DI SINI: Jika error-nya AUTH atau QUOTA, langsung keluar! 
            # Jangan buang-buang waktu menunggu jeda sleep.
            if error_type in ("auth", "quota") or attempt == max_retries:
                return None, error_type

            wait = base_delay * (2 ** (attempt - 1))
            logger.info(f"Menunggu {wait:.0f}s sebelum retry ke-{attempt + 1}...")
            time.sleep(wait)

    return None, "unknown"

def _format_context_as_text(normalized: List[Dict[str, Any]]) -> str:
    """Ubah normalized_context jadi teks teristruktur untuk prompt Gemini."""
    lines = []
    for i, item in enumerate(normalized, 1):
        parts = [f"{i}. {item['title']}"]
        if item["score"] is not None:
            parts.append(f"Rating {item['score']}%") # 🌟 REVISI: Ubah jadi persen
        if item["director"]:
            parts.append(f"Sutradara: {item['director']}")
        if item["genres"]:
            parts.append(f"Genre: {', '.join(item['genres'])}")
        if item["description"]:
            desc = item["description"][:200].strip()
            suffix = "..." if len(item["description"]) > 200 else ""
            parts.append(f'Sinopsis: "{desc}{suffix}"')
        lines.append(" | ".join(parts))
    return "\n".join(lines)

def _offline_format_answer(normalized: List[Dict[str, Any]], question: str) -> str:
    """Formatter offline Python (tanpa Gemini) sebagai jaring pengaman terakhir."""
    lines = [
        "Hei! Layanan AI teks kami lagi padat sebentar, tapi saya berhasil narik datanya "
        "langsung dari database. Berikut jawabannya:\n"
    ]
    for i, item in enumerate(normalized, 1):
        # Jalur jika yang keluar adalah data Sutradara
        if item['title'] == "Film Tanpa Judul" and item['director']:
            film_line = f"**{i}. {item['director']}**"
            details = []
            if item["genres"]:
                details.append(f"🎭 {', '.join(item['genres'])}")
            if item["score"] is not None:
                details.append(f"⭐ {item['score']}%") # 🌟 REVISI: Jadi persen
            if details:
                film_line += f"  —  {' | '.join(details)}"
        
        # Jalur standar jika yang keluar adalah daftar Film
        else:
            film_line = f"**{i}. {item['title']}**"
            details = []
            if item["score"] is not None:
                details.append(f"⭐ {item['score']}%") # 🌟 REVISI: Jadi persen
            if item["director"]:
                details.append(f"🎬 {item['director']}")
            if item["genres"]:
                details.append(f"🎭 {', '.join(item['genres'])}")
            if details:
                film_line += f"  —  {' | '.join(details)}"
            if item["description"]:
                desc_short = item["description"][:150].strip()
                film_line += f"\n   _{desc_short}{'...' if len(item['description']) > 150 else ''}_"
                
        lines.append(film_line)

    lines.append("\nSemoga membantu! Kalau mau cari yang lain, tanya aja ya 😊")
    return "\n\n".join(lines)


# ==========================================
# 📊 MODUL EVALUASI: PELACAK LATENCY OPTIMASI
# ==========================================
@contextmanager
def track_node_latency(node_name: str, reasoning_steps: list):
    """
    Context manager untuk menghitung waktu eksekusi setiap node RAG.
    Hasilnya akan langsung disuntikkan ke dalam list log frontend.
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Format teks log agar rapi di accordion web
        log_message = f"⏱️ [Latency] {node_name} selesai dieksekusi dalam {elapsed_ms:.2f} ms"
        reasoning_steps.append(log_message)
        logger.info(f"[PROFILER] {node_name}: {elapsed_ms:.2f} ms")