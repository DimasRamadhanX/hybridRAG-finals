import time
import logging
from django.core.management.base import BaseCommand
from chatbot.services.neo4j_client import driver
from chatbot.services.local_embedding_client import get_local_embedding  # 🌟 Menggunakan versi lokal Hugging Face

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Mengambil data film dari Neo4j, membuat embedding via Hugging Face secara offline, dan menyimpannya kembali ke node Film."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Memulai proses pembuatan embedding lokal (Hugging Face)..."))
        start_time = time.time()

        # 1. AMBIL DATA AWAL (Hanya mengambil film yang belum memiliki vektor)
        fetch_query = """
        MATCH (f:Film)
        WHERE f.embedding_vector IS NULL
        OPTIONAL MATCH (d:Director)-[:DIRECTED]->(f)
        OPTIONAL MATCH (f)-[:HAS_GENRE]->(g:Genre)
        RETURN elementId(f) AS id, f.title AS title, f.description AS description, 
               d.name AS director, collect(g.name) AS genres
        LIMIT 1400
        """

        self.stdout.write("Membaca data film dari Neo4j...")
        with driver.session() as read_session:
            try:
                records = read_session.run(fetch_query)
                movie_list = [record for record in records]
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"[FATAL] Gagal mengambil data awal dari Neo4j: {str(e)}"))
                return

        total_movies = len(movie_list)
        if total_movies == 0:
            self.stdout.write(self.style.SUCCESS("🎉 Semua data film di database Anda sudah memiliki embedding vector lokal!"))
            return

        self.stdout.write(self.style.SUCCESS(f"Ditemukan {total_movies} film yang perlu diproses pada batch ini."))

        # 2. PERULANGAN PROSES EMBEDDING (100% OFFLINE & MAKSIMAL)
        success_count = 0
        for index, movie in enumerate(movie_list, 1):
            movie_id = movie["id"]
            title = movie["title"] or "Film Tanpa Judul"
            description = movie["description"] or ""
            director = movie["director"] or "Unknown Director"
            genres = ", ".join(movie["genres"]) if movie["genres"] else "Unknown Genre"

            # Satukan komponen teks menjadi satu kesatuan informasi kaya makna semantik
            rich_text = f"Judul: {title}. Sutradara: {director}. Genre: {genres}. Sinopsis: {description}"

            try:
                # Dapatkan koordinat vektor (384 dimensi) dari model lokal Hugging Face
                vector = get_local_embedding(rich_text)

                if vector is None:
                    raise RuntimeError("Fungsi lokal mengembalikan nilai kosong (None).")

                # Buka sesi kilat untuk memperbarui properti node di Neo4j
                update_query = """
                MATCH (f:Film)
                WHERE elementId(f) = $id
                SET f.embedding_vector = $vector
                """
                with driver.session() as write_session:
                    write_session.run(update_query, id=movie_id, vector=vector)
                
                self.stdout.write(f"[{index}/{total_movies}] Sukses Vektor Lokal ➡️ {title}")
                success_count += 1

            except Exception as e:
                # Karena berjalan lokal, error di sini biasanya akibat masalah data corrupt atau koneksi DB terputus
                self.stdout.write(self.style.ERROR(f"❌ Gagal memproses film '{title}': {str(e)}"))
                continue  # Lewati film bermasalah, langsung lanjut ke antrean berikutnya

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.stdout.write(self.style.SUCCESS(
            f"\n🎉 Selesai! Berhasil memperbarui {success_count}/{total_movies} film "
            f"dalam waktu {elapsed_time:.2f} detik tanpa terkena rate limit."
        ))