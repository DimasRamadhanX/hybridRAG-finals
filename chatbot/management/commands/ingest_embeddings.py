import time
import re
import logging
from django.core.management.base import BaseCommand
from chatbot.services.neo4j_client import driver
from chatbot.services.gemini_client import get_embedding

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Mengambil data film dari Neo4j, membuat ragam embedding via Gemini, dan menyimpannya kembali ke node Film dengan proteksi error."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Memulai proses pembuatan embedding..."))

        # 1. AMBIL DATA AWAL (Gunakan sesi singkat untuk membaca)
        fetch_query = """
        MATCH (f:Film)
        WHERE f.embedding_vector IS NULL
        OPTIONAL MATCH (d:Director)-[:DIRECTED]->(f)
        OPTIONAL MATCH (f)-[:HAS_GENRE]->(g:Genre)
        RETURN elementId(f) AS id, f.title AS title, f.description AS description, 
               d.name AS director, collect(g.name) AS genres
        LIMIT 1400
        """

        self.stdout.write("Menghubungi database Neo4j untuk memeriksa kuota data...")
        with driver.session() as read_session:
            try:
                records = read_session.run(fetch_query)
                movie_list = [record for record in records]
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"[FATAL] Gagal mengambil data awal dari Neo4j: {str(e)}"))
                return

        total_movies = len(movie_list)
        if total_movies == 0:
            self.stdout.write(self.style.SUCCESS("🎉 Semua data film di database Anda sudah memiliki embedding vector!"))
            return

        self.stdout.write(self.style.SUCCESS(f"Ditemukan {total_movies} film yang perlu diproses pada batch ini."))

        # 2. PERULANGAN PROSES EMBEDDING
        for index, movie in enumerate(movie_list, 1):
            movie_id = movie["id"]
            title = movie["title"] or "Film Tanpa Judul"
            description = movie["description"] or ""
            director = movie["director"] or "Unknown Director"
            genres = ", ".join(movie["genres"]) if movie["genres"] else "Unknown Genre"

            # Satukan teks menjadi satu kesatuan informasi yang kaya (Rich Semantic Text)
            rich_text = f"Judul: {title}. Sutradara: {director}. Genre: {genres}. Sinopsis: {description}"

            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    # Dapatkan koordinat vektor dari Gemini
                    vector = get_embedding(rich_text)

                    # 🌟 FIX CELAH 1: Paksa lempar RuntimeError jika fungsi client mengembalikan None
                    if vector is None:
                        raise RuntimeError("Gemini API gagal menghasilkan vektor (kemungkinan RESOURCE_EXHAUSTED atau kuota habis)")

                    # 🌟 FIX CELAH 2: Buka sesi tulis mandiri (short-lived session) agar tidak menahan connection pool database
                    update_query = """
                    MATCH (f:Film)
                    WHERE elementId(f) = $id
                    SET f.embedding_vector = $vector
                    """
                    with driver.session() as write_session:
                        write_session.run(update_query, id=movie_id, vector=vector)
                    
                    self.stdout.write(f"[{index}/{total_movies}] Berhasil memperbarui vektor: {title}")
                    success = True
                    
                    # Jeda stabilitas request pada Free Tier
                    time.sleep(0.2)

                except Exception as e:
                    error_msg = str(e).lower()

                    # KATEGORI A: FAIL-FAST (Kredensial Salah / API Key Mati)
                    if "key not valid" in error_msg or "invalid_argument" in error_msg or "400" in error_msg:
                        self.stdout.write(self.style.ERROR(f"\n[FATAL] API Key Gemini ditolak atau tidak valid."))
                        self.stdout.write(self.style.ERROR(f"Detail: {str(e)}"))
                        self.stdout.write(self.style.WARNING("Proses dihentikan otomatis untuk menghemat resource."))
                        return

                    # KATEGORI B: RECOVERABLE ERROR (Rate Limit / Quota Habis Sementara)
                    if "429" in error_msg or "exhausted" in error_msg or "quota" in error_msg:
                        retry_count += 1
                        wait_time = 20 * retry_count
                        self.stdout.write(self.style.WARNING(
                            f"\n[RATE LIMIT] Server Gemini sibuk atau kuota terlampaui. "
                            f"Mencoba kembali dalam {wait_time} detik... (Percobaan {retry_count}/{max_retries})"
                        ))
                        time.sleep(wait_time)
                        continue

                    # KATEGORI C: UNKNOWN ERROR (Eror database atau format karakter khusus pada film tertentu)
                    self.stdout.write(self.style.ERROR(f"Gagal memproses film '{title}': {str(e)}"))
                    break  # Lewati film bermasalah ini, lanjut ke antrean berikutnya

        self.stdout.write(self.style.SUCCESS("\nSelesai! Seluruh data film pada batch ini berhasil dieksekusi."))