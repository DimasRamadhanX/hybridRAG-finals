import os
import json
import re
from django.core.management.base import BaseCommand
from chatbot.services.neo4j_client import driver
from chatbot.services.local_embedding_client import get_local_embedding
from chatbot.services.rag_service import call_gemini_with_retry 

class Command(BaseCommand):
    help = "Mengekstrak entitas dan relasi dari file teks tidak terstruktur mana pun untuk membangun Knowledge Graph."

    def add_arguments(self, parser):
        # ⚡ MENGUBAH MENJADI POSITIONAL ARGUMENT (WAJIB DIISI)
        parser.add_argument(
            'file_path', 
            type=str, 
            help='Path lengkap atau nama file .txt yang ingin ditransformasikan'
        )

    def handle(self, *args, **options):
        file_path = options['file_path']
        
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f"❌ File tidak ditemukan di jalur: {file_path}"))
            self.stdout.write(self.style.WARNING("💡 Pastikan file berada di folder yang sama atau tulis path lengkapnya."))
            return

        self.stdout.write(self.style.WARNING(f"📖 Membaca teks tidak terstruktur dari file: {file_path}..."))
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()

        if not raw_text:
            self.stdout.write(self.style.ERROR("❌ File tersebut kosong! Tidak ada teks untuk diproses."))
            return

        self.stdout.write(self.style.WARNING("🧠 Mengirim teks mentah ke Gemini untuk menganalisis skema grafik..."))
        
        # Prompt dibuat sepenuhnya dinamis agar bisa mengekstrak film apa saja secara akurat
        prompt = f"""
        Anda adalah pakar extraction informasi untuk Knowledge Graph database Neo4j.
        Tugas Anda adalah membaca teks narasi/sinopsis tidak terstruktur di bawah ini, 
        mendeteksi entitas Film, Sutradara (Director), serta Genre, lalu merakitnya ke dalam format JSON.

        Teks Mentah:
        "{raw_text}"

        Format JSON yang WAJIB Anda kembalikan:
        {{
            "title": "Nama Judul Film yang ditemukan",
            "description": "Ringkasan sinopsis pendek objek film tersebut",
            "score": 80, // Berikan estimasi rating skala 0-100 berdasarkan bobot ulasan di dalam teks kalimatnya
            "director": "Nama Sutradara yang terdeteksi (tulis null jika tidak ada)",
            "genres": ["Genre1", "Genre2"] // List genre dalam bahasa Inggris (contoh: Sci-Fi, Horror, Drama, Action)
        }}

        Ketentuan Ekstraksi:
        - Berikan HANYA respons berupa JSON murni.
        - JANGAN sertakan markdown seperti ```json, jangan ada kata 'json', dan tanpa teks penjelasan apa pun.
        """

        response_text, error_type = call_gemini_with_retry(prompt)
        
        if not response_text:
            self.stdout.write(self.style.ERROR(f"❌ LLM gagal merespons atau terkena limit kuota: {error_type}"))
            return

        clean_json_str = re.sub(r'```[a-zA-Z]*\n?|```', '', response_text).strip()

        try:
            extracted_data = json.loads(clean_json_str)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Gagal mengubah respons teks LLM menjadi format objek JSON: {str(e)}"))
            self.stdout.write(f"Teks asli dari LLM: {response_text}")
            return

        self.stdout.write(self.style.SUCCESS("✨ Ekstraksi LLM Berhasil! Hasil data struktur baru:"))
        self.stdout.write(json.dumps(extracted_data, indent=4))

        # Ambil data hasil parsing
        title = extracted_data.get("title")
        description = extracted_data.get("description", "")
        score = extracted_data.get("score", 70)
        director_name = extracted_data.get("director")
        genres = extracted_data.get("genres", [])

        if not title or title.lower() == "null":
            self.stdout.write(self.style.ERROR("❌ LLM gagal mendeteksi judul film dari teks tersebut. Proses dihentikan."))
            return

        # --- TAHAP MEMASUKKAN DATA KE NEO4J VIA GRAPH BUILDER ---
        self.stdout.write(self.style.WARNING(f"🖥️ Menyuntikkan entitas grafik film '{title}' ke Neo4j..."))
        
        cypher_query = """
        MERGE (f:Film {title: $title})
        SET f.description = $description,
            f.score = toInteger($score)

        FOREACH (_ IN CASE WHEN $director IS NOT NULL AND $director <> 'null' AND $director <> '' THEN [1] ELSE [] END |
            MERGE (d:Director {name: $director})
            MERGE (d)-[:DIRECTED]->(f)
        )

        FOREACH (genre_name IN $genres |
            MERGE (g:Genre {name: genre_name})
            MERGE (f)-[:HAS_GENRE]->(g)
        )
        """

        with driver.session() as session:
            try:
                session.run(
                    cypher_query, 
                    title=title, 
                    description=description, 
                    score=score, 
                    director=director_name, 
                    genres=genres
                )
                self.stdout.write(self.style.SUCCESS(f"✅ Node Film '{title}' beserta hubungannya sukses dibangun!"))
            except Exception as db_err:
                self.stdout.write(self.style.ERROR(f"❌ Gagal memperbarui database Neo4j: {str(db_err)}"))
                return

        # --- TAHAP AUTO EMBEDDING SEMANTIK ---
        self.stdout.write(self.style.WARNING("🔮 Memperbarui indeks vektor untuk pencarian semantik..."))
        
        # Pastikan Vector Index 'movie_vector_index' ada di Neo4j
        create_index_query = """
        CREATE VECTOR INDEX movie_vector_index IF NOT EXISTS
        FOR (f:Film)
        ON (f.embedding_vector)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
          }
        }
        """
        with driver.session() as session:
            try:
                session.run(create_index_query)
            except Exception as index_err:
                self.stdout.write(self.style.WARNING(f"⚠️ Peringatan saat memastikan index: {str(index_err)}"))

        try:
            genres_str = ", ".join(genres) if genres else "Unknown Genre"
            rich_text = f"Judul: {title}. Sutradara: {director_name}. Genre: {genres_str}. Sinopsis: {description}"
            
            raw_vector = get_local_embedding(rich_text)
            
            if hasattr(raw_vector, "tolist"):
                vector_data = raw_vector.tolist()
            else:
                vector_data = raw_vector

            vector_query = """
            MATCH (f:Film {title: $title})
            SET f.embedding_vector = $vector
            """
            with driver.session() as session:
                session.run(vector_query, title=title, vector=vector_data)
            self.stdout.write(self.style.SUCCESS(f"🎉 Selesai Sempurna! Film '{title}' sekarang sudah bisa dicari di chatbot."))
            
        except Exception as embed_err:
            self.stdout.write(self.style.ERROR(f"⚠️ Gagal membuat koordinat vektor otomatis: {str(embed_err)}"))

        self.stdout.write(self.style.SUCCESS(f"\n🎯 Proses Transformasi File '{file_path}' Selesai Terbuka!"))