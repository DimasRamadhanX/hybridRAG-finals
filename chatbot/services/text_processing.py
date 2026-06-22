import re
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Tuple, Optional

# --- Peta genre: semua varian Indonesia/Inggris/typo → format Wikidata ---
_GENRE_VARIANTS: Dict[str, str] = {
    # Komedi
    "komedi": "comedy film", "comedy": "comedy film", "komedy": "comedy film",
    "komidi": "comedy film", "komdie": "comedy film", "lucu": "comedy film",
    # Aksi
    "aksi": "action film", "action": "action film", "aksion": "action film",
    "actin": "action film", "laga": "action film", "actoin": "action film",
    # Romantis
    "romantis": "romance film", "roman": "romance film", "romansa": "romance film",
    "romance": "romance film", "romatis": "romance film", "romantiss": "romance film",
    "cinta": "romance film",
    # Horor
    "horor": "horror film", "horror": "horror film", "horo": "horror film",
    "seram": "horror film", "horr": "horror film",
    # Fiksi Ilmiah
    "fiksi ilmiah": "science fiction film", "sci-fi": "science fiction film",
    "scifi": "science fiction film", "science fiction": "science fiction film",
    "fiksi": "science fiction film", "fiksi ilmia": "science fiction film",
    # Animasi
    "animasi": "animated film", "anime": "animated film", "kartun": "animated film",
    "animation": "animated film",
    # Drama
    "drama": "drama film", "drame": "drama film", "draama": "drama film",
    # Thriller
    "thriller": "thriller film", "triler": "thriller film", "thiller": "thriller film",
    "triller": "thriller film",
    # Dokumenter
    "dokumenter": "documentary film", "dokumentari": "documentary film",
    "documentary": "documentary film", "documenter": "documentary film",
    # Petualangan
    "petualangan": "adventure film", "adventure": "adventure film",
    "adventur": "adventure film", "petualagan": "adventure film",
    # Fantasi
    "fantasi": "fantasy film", "fantasy": "fantasy film",
    "fantasai": "fantasy film", "fantazy": "fantasy film",
    # Misteri
    "misteri": "mystery film", "mystery": "mystery film", "mistri": "mystery film",
    "miteri": "mystery film", "mistery": "mystery film",
    # Kriminal
    "kriminal": "crime film", "crime": "crime film", "krime": "crime film",
    "criminal": "crime film", "krimi": "crime film",
    # Musikal
    "musikal": "musical film", "musical": "musical film", "musik": "musical film",
    # Biografi
    "biografi": "biographical film", "biopic": "biographical film",
    "biogrfi": "biographical film",
    # Sejarah
    "sejarah": "historical film", "historical": "historical film",
    "history": "historical film", "sejarh": "historical film",
    # Keluarga
    "keluarga": "family film", "family": "family film", "anak": "family film",
}

# --- Peta kata umum: typo Indonesia umum pada pertanyaan film ---
_WORD_CORRECTIONS: Dict[str, str] = {
    "rekomendasiin": "rekomendasi", "rekomendasi": "rekomendasi", "rekomen": "rekomendasi",
    "recomendasi": "rekomendasi", "rekomndasi": "rekomendasi",
    "cariin": "cari", "carikan": "cari", "cari": "cari",
    "tunjukin": "tampilkan", "tampilkan": "tampilkan", "kasih tau": "tampilkan",
    "sutradra": "sutradara", "sutradar": "sutradara", "directur": "sutradara",
    "sutardara": "sutradara", "direktur": "sutradara",
    "flim": "film", "fillm": "film", "filem": "film", "pilm": "film",
    "raiting": "rating", "ratting": "rating", "rting": "rating",
    "sinopsiss": "sinopsis", "sinopssis": "sinopsis", "synopsis": "sinopsis",
    "terbaik": "terbaik", "terbai": "terbaik", "terbyak": "terbaik",
    "populer": "populer", "popuper": "populer", "popular": "populer",
    "terpopuler": "terpopuler", "ter populer": "terpopuler",
}

def _str_similarity(a: str, b: str) -> float:
    """Hitung similarity ratio antara dua string (case-insensitive)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def _find_best_genre_match(word: str, threshold: float = 0.85) -> Tuple[Optional[str], float]:
    """
    Cari genre terbaik dari _GENRE_VARIANTS berdasarkan similarity.
    🌟 REVISI: Threshold dinaikkan ke 0.85 agar kata aman tidak dituduh typo.
    """
    word_lower = word.lower().strip()

    if word_lower in _GENRE_VARIANTS:
        return _GENRE_VARIANTS[word_lower], 1.0

    best_score = 0.0
    best_key = None
    for key in _GENRE_VARIANTS:
        score = _str_similarity(word_lower, key)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= threshold and best_key:
        return _GENRE_VARIANTS[best_key], best_score

    return None, best_score

def _correct_single_word(word: str, threshold: float = 0.85) -> Tuple[str, bool]:
    """Koreksi satu kata menggunakan _WORD_CORRECTIONS."""
    word_lower = word.lower().strip()
    if word_lower in _WORD_CORRECTIONS:
        return _WORD_CORRECTIONS[word_lower], True

    keys = list(_WORD_CORRECTIONS.keys())
    matches = get_close_matches(word_lower, keys, n=1, cutoff=threshold)
    if matches:
        corrected = _WORD_CORRECTIONS[matches[0]]
        return corrected, True

    return word, False

def _extract_and_correct_genre_phrases(text: str, threshold: float = 0.85) -> Tuple[str, List[str]]:
    """
    Scan teks untuk frasa genre dan menerjemahkannya menggunakan batas kata aman (\\b).
    🌟 REVISI: Menghapus hardcoded 0.72 dan menyinkronkannya dengan variabel threshold utama.
    """
    corrections = []
    result = text

    words = text.lower().split()
    i = 0

    while i < len(words):
        matched = False
        for n in (3, 2, 1):
            if i + n > len(words):
                continue
            phrase = " ".join(words[i:i+n])
            
            # Panggil fungsi pencari dengan batas threshold yang ketat
            genre_match, score = _find_best_genre_match(phrase, threshold=threshold)
            if genre_match:
                original_phrase = " ".join(text.split()[i:i+n])
                if original_phrase.lower() != genre_match.lower():
                    pattern = re.compile(fr"\b{re.escape(original_phrase)}\b", re.IGNORECASE)
                    result = pattern.sub(genre_match, result, count=1)
                    corrections.append(
                        f'"{original_phrase}" → "{genre_match}"'
                        + (f" (koreksi typo, sim={score:.0%})" if score < 1.0 else " (terjemahan)")
                    )
                i += n
                matched = True
                break
        if not matched:
            i += 1

    return result, corrections

def _correct_common_words(text: str) -> Tuple[str, List[str]]:
    """Koreksi kata-kata umum dengan mempertahankan tanda baca depan & belakang secara aman."""
    corrections = []
    words = text.split()
    corrected_words = []

    for word in words:
        stripped = word.strip("?!.,;:\"'()")
        
        if not stripped:
            corrected_words.append(word)
            continue
            
        corrected, was_corrected = _correct_single_word(stripped)
        
        if was_corrected and corrected != stripped.lower():
            leading_idx = word.find(stripped)
            leading_punc = word[:leading_idx]
            trailing_punc = word[leading_idx + len(stripped):]
            
            corrected_words.append(leading_punc + corrected + trailing_punc)
            corrections.append(f'"{stripped}" → "{corrected}"')
        else:
            corrected_words.append(word)

    return " ".join(corrected_words), corrections

def preprocess_user_input(raw_question: str) -> Tuple[str, List[str]]:
    """Pipeline normalisasi & koreksi typo lengkap untuk input pengguna."""
    all_corrections: List[str] = []

    text = re.sub(r'\s+', ' ', raw_question).strip()
    text, word_corrections = _correct_common_words(text)
    all_corrections.extend(word_corrections)

    # 🌟 Melemparkan filter threshold ketat ke pemrosesan frasa genre
    text, genre_corrections = _extract_and_correct_genre_phrases(text, threshold=0.85)
    all_corrections.extend(genre_corrections)

    return text, all_corrections