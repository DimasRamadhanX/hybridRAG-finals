import re
import logging
from typing import TypedDict, List, Dict, Any, Tuple, Optional
from chatbot.services.neo4j_client import driver
from chatbot.services.text_processing import _GENRE_VARIANTS, preprocess_user_input
from chatbot.services.local_embedding_client import get_local_embedding  # 🌟 Menggunakan Hugging Face Lokal
from chatbot.services.rag_utils import (
    normalize_graph_context,
    call_gemini_with_retry,
    _format_context_as_text,
    _offline_format_answer,
    track_node_latency,
)

logger = logging.getLogger(__name__)

# ==========================================
# 1. SKEMA DATABASE KNOWLEDGE GRAPH
# ==========================================
NEO4J_SCHEMA = """
Nodes dan Properti:
1. (:Film {title: STRING, description: STRING, score: INTEGER, embedding_vector: LIST})
   *Catatan penting: Properti 'score' disimpan dalam skala angka 0 sampai 100 (Contoh: rating 8/10 ditulis sebagai 80, rating 7.5 ditulis 75).
2. (:Director {name: STRING, description: STRING})
3. (:Genre {name: STRING, description: STRING})

Relationships (Hubungan):
1. (:Director)-[:DIRECTED]->(:Film)
2. (:Film)-[:HAS_GENRE]->(:Genre)

Indeks Vektor yang Tersedia:
- Nama indeks: 'film_embeddings_idx'
- Cara panggil untuk pencarian semantik teks: 
  CALL db.index.vector.queryNodes('film_embeddings_idx', $k, $vector) YIELD node AS f, score
"""

# ==========================================
# 2. DEFINISI STATE
# ==========================================
class GraphRAGState(TypedDict):
    user_question: str          # Pertanyaan ASLI dari user
    cleaned_question: str       # Pertanyaan setelah koreksi typo & normalisasi
    history: List[Dict[str, str]]
    cypher_query: str
    cypher_attempts: int
    graph_context: List[Dict[str, Any]]
    normalized_context: List[Dict[str, Any]]
    embedding_error_occurred: bool
    api_error_type: Optional[str]   # "quota" | "auth" | "network" | "unknown"
    correction_notes: List[str]     # Log koreksi typo yang dilakukan
    reasoning_steps: List[str]
    final_answer: str


# ==========================================
# 3. NODES
# ==========================================

def node_preprocess_input(state: GraphRAGState) -> GraphRAGState:
    """Node 0: Normalisasi & koreksi typo input pengguna sebelum masuk pipeline."""
    cleaned, corrections = preprocess_user_input(state["user_question"])
    state["cleaned_question"] = cleaned
    state["correction_notes"] = corrections

    if corrections:
        notes_str = ", ".join(corrections)
        state["reasoning_steps"].append(f"✏️ Input dikoreksi/dinormalisasi: {notes_str}")
        logger.info(f"[Preprocessor] '{state['user_question']}' → '{cleaned}' | koreksi: {corrections}")
    else:
        state["reasoning_steps"].append("✅ Input bersih, tidak ada koreksi diperlukan.")

    return state


def node_generate_cypher_agent(state: GraphRAGState, previous_failed_query: str = "") -> GraphRAGState:
    """
    Node 1: Generate query Cypher via Gemini Flash.
    Menggunakan cleaned_question (sudah dikoreksi typo & genre diterjemahkan).
    Mendukung self-correction serta HARDCODED LOKAL FALLBACK jika kuota API habis.
    Ditambahkan: Fitur Command Interceptor untuk trigger -rankFilm GDS dan Tameng Anti-Halusinasi.
    """
    original_q = state["user_question"]

    # ⚡ 1. COMMAND INTERCEPTOR: Pemicu Rahasia -rankFilm GDS
    if "-rankfilm" in original_q.lower():
        state["cypher_query"] = """
            MATCH (f:Film) 
            WHERE f.pagerank_score IS NOT NULL 
            RETURN f.title AS title, f.pagerank_score AS score 
            ORDER BY f.pagerank_score DESC 
            LIMIT 5
        """.strip()
        state["reasoning_steps"].append("⚡ [Command Trigger] Mendeteksi '-rankFilm'. Menyuntikkan kueri urutan PageRank GDS.")
        return state

    # --- 2. JALUR NORMAL (JIKA BUKAN TRIGGER COMMAND) ---
    attempt_num = state.get("cypher_attempts", 0) + 1
    state["reasoning_steps"].append(f"🧠 Merancang query Cypher (percobaan ke-{attempt_num})...")

    correction_hint = ""
    if previous_failed_query:
        correction_hint = f"""
        ⚠️ Query sebelumnya menghasilkan nol baris data:
        ```
        {previous_failed_query}
        ```
        Tolong buat versi lebih longgar: hilangkan filter ketat, gunakan CONTAINS bukan =,
        kurangi kondisi WHERE, atau gunakan pencarian vektor sebagai alternatif.
        """

    question_for_cypher = state.get("cleaned_question") or state["user_question"]

    prompt = f"""
    Anda adalah pakar database Neo4j Cypher. Ubah pertanyaan pengguna menjadi query Cypher valid
    sesuai skema berikut:

    {NEO4J_SCHEMA}

    Aturan Krusial:
    1. Berikan HANYA query Cypher saja, tanpa penjelasan, tanpa backticks (```), tanpa kata 'cypher'.
    2. Genre di database sudah dalam format Bahasa Inggris Wikidata — pertanyaan pengguna sudah 
       diterjemahkan oleh sistem sebelum sampai ke sini, jadi gunakan apa adanya.
    3. SELALU gunakan `toLower(properti) CONTAINS toLower("nilai")` untuk filter teks 
       (judul, genre, nama sutradara) — JANGAN gunakan operator `=` untuk string.
    4. Untuk RATING/SKOR, konversikan ke skala 0-100 
       (misal: "sekitar 8" → WHERE f.score BETWEEN 75 AND 85).
    5. Gunakan parameter `$vector` HANYA untuk pertanyaan semantik tentang sinopsis/cerita.
    6. Khusus pencarian vektor (queryNodes), Anda WAJIB menggunakan nama indeks 'movie_vector_index' 
       dan langsung menuliskan angka batasan (seperti 5 atau 10) pada argumen kedua. 
       JANGAN PERNAH menggunakan variabel seperti $k atau $limit karena akan membuat sistem crash.
       (Contoh wajib: CALL db.index.vector.queryNodes('movie_vector_index', 10, $vector) YIELD node AS f, score AS similarity_score).
    7. Selalu sertakan LIMIT (maksimal 10) di akhir kueri, kecuali untuk kueri agregasi atau kueri 
       vektor (queryNodes) yang jumlah datanya sudah dibatasi di dalam prosedur internalnya.
    8. SELALU gunakan alias 'AS' yang eksplisit pada bagian RETURN agar format data tidak hilang 
       saat dibaca komponen Python (Contoh wajib: f.title AS title, d.name AS director, f.score AS score).

    {correction_hint}

    Pertanyaan Pengguna (sudah dinormalisasi): "{question_for_cypher}"
    Query Cypher:
    """

    text, error_type = call_gemini_with_retry(prompt)
    state["cypher_attempts"] = state.get("cypher_attempts", 0) + 1

    if text:
        # Bersihkan format markdown backticks bawaan LLM
        cleaned = re.sub(r'```[a-zA-Z]*\n?|```', '', text).strip()
        
        # 🛡️ TAMENG ANTI-HALUSINASI JALUR PROGRAMMATIK
        # Jika Gemini ngeyel menulis nama indeks yang salah, kita paksa ganti di sini
        if "film_embeddings_idx" in cleaned:
            cleaned = cleaned.replace("film_embeddings_idx", "movie_vector_index")
            state["reasoning_steps"].append("🛡️ [Guardrail] Memperbaiki halusinasi nama indeks secara otomatis.")
            
        state["cypher_query"] = cleaned
        state["reasoning_steps"].append(f"🤖 Query dirancang: `{cleaned}`")
    else:
        # 🌟 JARING PENGAMAN KETIKA GEMINI TERKENA LIMIT KUOTA (429)
        state["api_error_type"] = error_type
        state["reasoning_steps"].append(f"⚠️ Gagal merancang query ({error_type}). Mengaktifkan Rule-Based Fallback...")
        
        cleaned_q = (state.get("cleaned_question") or state["user_question"]).lower()
        
        # 1. Ekstrak kata kunci berhuruf kapital (indikasi nama sutradara/aktor/judul)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', original_q)
        
        # Singkirkan kata umum yang kebetulan kapital di awal kalimat agar tidak dikira nama orang
        exclusions = {"Halo", "Hai", "Cari", "Carikan", "Rekomendasi", "Tampilkan", "Film", "Sutradara", "Genre"}
        potential_names = [word for word in capitalized_words if word not in exclusions]
        
        # Gabungkan kata nama yang tersisa (misal: "Alan", "Taylor" -> "Alan Taylor")
        full_name_match = " ".join(potential_names).strip()
        
        # 2. Cari tahu apakah ada genre hasil terjemahan Wikidata di dalam teks pertanyaan
        found_genre = None
        for g_wikidata in set(_GENRE_VARIANTS.values()):
            if g_wikidata in cleaned_q:
                found_genre = g_wikidata
                break
        
        # KONDISI A: Jika terdeteksi Nama Orang DAN ada Genre
        if full_name_match and found_genre:
            state["cypher_query"] = f"MATCH (d:Director)-[:DIRECTED]->(f:Film)-[:HAS_GENRE]->(g:Genre) WHERE toLower(d.name) CONTAINS toLower('{full_name_match}') AND toLower(g.name) CONTAINS '{found_genre}' RETURN f.title AS title, f.score AS score LIMIT 10"
            state["reasoning_steps"].append(f"🚨 [Lokal Fallback] Menyuntikkan kueri hibrida Sutradara + Genre: `{state['cypher_query']}`")
            
        # KONDISI B: Jika hanya terdeteksi Nama Orang (Sutradara)
        elif full_name_match:
            state["cypher_query"] = f"MATCH (d:Director)-[:DIRECTED]->(f:Film) WHERE toLower(d.name) CONTAINS toLower('{full_name_match}') RETURN f.title AS title, f.score AS score LIMIT 10"
            state["reasoning_steps"].append(f"🚨 [Lokal Fallback] Menyuntikkan kueri pencarian sutradara otomatis: `{state['cypher_query']}`")
            
        # KONDISI C: Jika hanya terdeteksi Genre saja
        elif found_genre:
            state["cypher_query"] = f"MATCH (f:Film)-[:HAS_GENRE]->(g:Genre) WHERE toLower(g.name) CONTAINS '{found_genre}' RETURN f.title AS title, f.score AS score LIMIT 10"
            state["reasoning_steps"].append(f"🚨 [Lokal Fallback] Menyuntikkan kueri teks genre otomatis: `{state['cypher_query']}`")
            
        # KONDISI D: Jalur alternatif standar rating tertinggi jika tidak ada entitas terdeteksi
        else:
            state["cypher_query"] = "MATCH (f:Film) WHERE f.score IS NOT NULL RETURN f.title AS title, f.score AS score ORDER BY f.score DESC LIMIT 5"
            state["reasoning_steps"].append("🚨 [Lokal Fallback] Menyuntikkan query daftar film rating tertinggi.")

    return state

def node_execute_graph_query(state: GraphRAGState) -> GraphRAGState:
    """Node 2: Eksekusi query Cypher ke Neo4j."""
    if not state["cypher_query"]:
        state["graph_context"] = []
        return state

    state["reasoning_steps"].append("🖥️ Mengirim query ke Neo4j...")

    parameters = {}
    if "$vector" in state["cypher_query"]:
        state["reasoning_steps"].append("🔮 Query semantik terdeteksi, mengambil embedding lokal...")
        embed_source = state.get("cleaned_question") or state["user_question"]
        try:
            # Ambil hasil array mentah dari Hugging Face lokal
            raw_vector = get_local_embedding(embed_source)
            
            # ⚡ JARING PENGAMAN UTAMA: Paksa konversi dari Numpy Array ke List biasa Python
            # Ini untuk menyembuhkan eror Neo.ClientError.Procedure.ProcedureCallFailed
            if hasattr(raw_vector, "tolist"):
                parameters["vector"] = raw_vector.tolist()
            else:
                parameters["vector"] = raw_vector
                
            parameters["k"] = 5
            state["embedding_error_occurred"] = False
        except Exception as e:
            state["embedding_error_occurred"] = True
            state["api_error_type"] = "unknown"
            state["graph_context"] = []
            state["normalized_context"] = []
            state["reasoning_steps"].append(f"💥 Gagal mengambil embedding lokal: {str(e)}")
            return state

    with driver.session() as session:
        try:
            results = session.run(state["cypher_query"], **parameters)
            raw = [record.data() for record in results]
            state["graph_context"] = raw
            state["normalized_context"] = normalize_graph_context(raw)
            state["reasoning_steps"].append(f"📊 Neo4j mengembalikan {len(raw)} record.")
        except Exception as e:
            logger.error(f"[Neo4j Error]: {e} | Query: {state['cypher_query']}")
            state["graph_context"] = []
            state["normalized_context"] = []
            state["reasoning_steps"].append(f"❌ Query gagal dieksekusi Neo4j: {str(e)[:120]}")

    return state

def node_self_correct_cypher(state: GraphRAGState) -> GraphRAGState:
    """Node 2b: Self-correction jika query pertama kosong."""
    state["reasoning_steps"].append("🔄 Hasil kosong. Mencoba koreksi query otomatis...")
    failed_query = state["cypher_query"]
    state = node_generate_cypher_agent(state, previous_failed_query=failed_query)
    state = node_execute_graph_query(state)
    return state


def node_fallback_semantic_gate(state: GraphRAGState) -> GraphRAGState:
    """Node 3: Fallback via vector search jika jalur Cypher kosong."""
    state["reasoning_steps"].append("⚠️ Semua jalur Cypher kosong. Mengaktifkan pencarian vektor semantik...")

    if state.get("embedding_error_occurred"):
        state["reasoning_steps"].append("⏭️ Lewati fallback: embedding sudah error sebelumnya.")
        return state

    embed_source = state.get("cleaned_question") or state["user_question"]
    try:
        # Menggunakan pustaka lokal Hugging Face (384 dimensi)
        query_vector = get_local_embedding(embed_source)
        state["embedding_error_occurred"] = False
    except Exception as e:
        state["embedding_error_occurred"] = True
        state["api_error_type"] = "unknown"
        state["reasoning_steps"].append(f"💥 Fallback gagal: embedding lokal error ({str(e)}).")
        return state

    fallback_query = """
    CALL db.index.vector.queryNodes('movie_vector_index', 5, $vector)
    YIELD node AS f, score AS vector_score
    WHERE vector_score > 0.40
    OPTIONAL MATCH (d:Director)-[:DIRECTED]->(f)
    OPTIONAL MATCH (f)-[:HAS_GENRE]->(g:Genre)
    // ⚡ Pastikan f.score dikembalikan SEBAGAI 'score' agar dibaca sistem normalisasi
    RETURN f.title AS title, f.description AS description, f.score AS score,
           d.name AS director, collect(g.name) AS genres, vector_score
    ORDER BY vector_score DESC
    """
    
    with driver.session() as session:
        try:
            results = session.run(fallback_query, vector=query_vector)
            raw = [record.data() for record in results]
            state["graph_context"] = raw
            state["normalized_context"] = normalize_graph_context(raw)
            state["reasoning_steps"].append(f"✅ Fallback vektor mendapat {len(raw)} film relevan.")
        except Exception as e:
            state["graph_context"] = []
            state["normalized_context"] = []
            state["reasoning_steps"].append(f"❌ Fallback query Neo4j gagal: {str(e)[:80]}")

    return state

def node_generate_answer(state: GraphRAGState) -> GraphRAGState:
    """Node 4: Sintesis jawaban akhir berdasarkan konteks yang terkumpul."""
    state["reasoning_steps"].append("✍️ Menyusun jawaban akhir...")

    normalized = state.get("normalized_context") or []

    if not normalized and state.get("embedding_error_occurred"):
        error_type = state.get("api_error_type", "unknown")
        messages = {
            "quota": (
                "Waduh, kuota API harian kami udah habis (Error 429) dan pencarian data juga "
                "nggak nemuin hasil. Coba tanyakan hal lain seperti nama sutradara tertentu ya!"
            ),
            "auth": (
                "Sepertinya ada masalah autentikasi dengan API kami saat ini. "
                "Tim teknis sudah diberitahu. Coba lagi dalam beberapa saat ya!"
            ),
        }
        state["final_answer"] = messages.get(error_type, "Ada gangguan koneksi ke layanan AI kami. Coba lagi nanti ya!")
        return state

    if not normalized:
        state["final_answer"] = (
            "Hmm, saya udah cari di seluruh database tapi nggak nemu film yang cocok "
            "dengan kriteria itu. Coba pertanyaannya diperluas atau gunakan kata kunci lain ya!"
        )
        return state

    context_text = _format_context_as_text(normalized)

    history_text = ""
    if state.get("history"):
        history_lines = []
        for turn in state["history"][-6:]:
            role = "Pengguna" if turn.get("role") == "user" else "Asisten"
            history_lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n".join(history_lines)

    correction_context = ""
    if state.get("correction_notes"):
        corrections_str = "; ".join(state["correction_notes"])
        correction_context = (
            " \n[CATATAN SISTEM: Input pengguna telah dikoreksi otomatis —  " + corrections_str + ". "
            "Pertanyaan asli: \"" + state['user_question'] + "\". "
            "Jika ada koreksi penting (terutama typo nama/judul), sebutkan dengan ramah di jawaban.]\n"
        )

    # ⚡ PERBAIKAN UTAMA: Menambahkan Aturan Vektor Semantik yang Tegas
    prompt = f"""
    Anda adalah Movie Assistant AI yang ramah, cerdas, dan berbicara santai khas anak muda Indonesia.

    Panduan gaya menjawab:
    1. Gunakan Bahasa Indonesia kasual ("kamu/aku"), hindari terlalu formal.
    2. Beri 1-2 kalimat pembuka yang relevan dengan topik (misal tentang baper, plot twist, dll).
    3. [PENTING] Jika ada data film di bagian [DATA FILM DARI DATABASE], data tersebut ADALAH film-film hasil 
       pencarian vektor semantik terbaik dari database grafik yang paling mendekati keinginan pengguna. 
       Anda WAJIB menampilkan, merekomendasikan, dan mengulas film-film tersebut! JANGAN PERNAH memberikan 
       jawaban penolakan seperti "tidak menemukan film" atau "tidak ada informasi" jika datanya tersedia.
    4. Sajikan detail rating, sutradara, genre secara natural — bukan sekadar bullet list kering.
    5. Tutup dengan kalimat singkat yang mengundang pertanyaan lanjutan.
    6. JANGAN mengarang fakta baru (seperti tahun rilis atau nama kru) di luar data yang diberikan.
    7. Jika ada koreksi typo penting (nama sutradara/judul film), sampaikan dengan ramah
       misal: "Btw, aku asumsiin maksudmu 'Christopher Nolan' ya 😊"

    {correction_context}
    {'[RIWAYAT PERCAKAPAN]' + chr(10) + history_text + chr(10) if history_text else ''}
    [DATA FILM DARI DATABASE]
    {context_text}

    [PERTANYAAN PENGGUNA]
    {state['user_question']}

    Jawaban:
    """

    text, error_type = call_gemini_with_retry(prompt)

    if text:
        state["final_answer"] = text.strip()
        state["reasoning_steps"].append("🎉 Jawaban berhasil disusun oleh Gemini.")
    else:
        state["api_error_type"] = error_type
        state["reasoning_steps"].append(f"⚠️ Gemini tidak tersedia ({error_type}). Menggunakan formatter offline...")
        state["final_answer"] = _offline_format_answer(normalized, state["user_question"])
        state["reasoning_steps"].append("🎉 Jawaban diformat secara lokal oleh Python.")

    return state


# ==========================================
# 4. EDGES & ROUTER
# ==========================================

def edge_router_evaluate_results(state: GraphRAGState) -> str:
    """Tentukan langkah selanjutnya berdasarkan hasil query."""
    has_data = bool(state.get("graph_context"))
    used_vector = "film_embeddings_idx" in state.get("cypher_query", "")
    embedding_failed = state.get("embedding_error_occurred", False)
    attempts = state.get("cypher_attempts", 1)

    if has_data:
        return "trigger_answer"
    if not used_vector and attempts <= 1:
        return "trigger_self_correct"
    if not embedding_failed:
        return "trigger_fallback"
    return "trigger_answer"


# ==========================================
# 5. ORKESTRASI PIPELINE (LATENCY OPTIMIZED)
# ==========================================

def generate_multi_hop_answer(
    user_question: str,
    history: List[Dict[str, str]] = None
) -> Tuple[str, List[str]]:
    """Pipeline utama GraphRAG dengan pemantauan profiler waktu dari utils."""
    state: GraphRAGState = {
        "user_question": user_question,
        "cleaned_question": "",
        "history": history or [],
        "cypher_query": "",
        "cypher_attempts": 0,
        "graph_context": [],
        "normalized_context": [],
        "embedding_error_occurred": False,
        "api_error_type": None,
        "correction_notes": [],
        "reasoning_steps": [],
        "final_answer": "",
    }

    # ⏱️ Mengukur Node 0: Koreksi Typo & Terjemahan
    with track_node_latency("Node 0: Input Preprocessing", state["reasoning_steps"]):
        state = node_preprocess_input(state)
        
    # ⏱️ Mengukur Node 1: Pembuat Rancangan Perintah Cypher
    with track_node_latency("Node 1: AI Cypher Agent", state["reasoning_steps"]):
        state = node_generate_cypher_agent(state)
        
    # ⏱️ Mengukur Node 2: Penarikan Data dari Database Grafik Neo4j
    with track_node_latency("Node 2: Neo4j Graph Execution", state["reasoning_steps"]):
        state = node_execute_graph_query(state)

    next_action = edge_router_evaluate_results(state)

    # ⏱️ Mengukur Jalur Percabangan Self-Correction jika Dipicu
    if next_action == "trigger_self_correct":
        with track_node_latency("Node 2b: Cypher Self-Correction Loop", state["reasoning_steps"]):
            state = node_self_correct_cypher(state)
        next_action = edge_router_evaluate_results(state)

    # ⏱️ Mengukur Jalur Percabangan Fallback Vektor jika Dipicu
    if next_action == "trigger_fallback":
        with track_node_latency("Node 3: Vector Semantic Fallback", state["reasoning_steps"]):
            state = node_fallback_semantic_gate(state)

    # ⏱️ Mengukur Node 4: Perangkuman Jawaban Akhir oleh AI
    with track_node_latency("Node 4: Final Synthesis & Generation", state["reasoning_steps"]):
        state = node_generate_answer(state)

    return state["final_answer"], state["reasoning_steps"]