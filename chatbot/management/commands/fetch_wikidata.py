import re
import time
import requests
import logging
from django.core.management.base import BaseCommand
from chatbot.services.neo4j_client import driver

logger = logging.getLogger(__name__)

# --- 1. UTILITY: FUNGSI NORMALISASI SKOR (DARI MAIN.PY FASTAPI) ---
def normalize_to_100(raw_score):
    """
    Mengonversi berbagai format teks skor (8.5/10, 85%, 4/5, dll) 
    menjadi angka integer murni skala 0-100.
    """
    if not raw_score or str(raw_score).strip() in ['No score', 'Unknown', '', 'None', 'nan']:
        return 0

    text = str(raw_score).lower().strip()

    # 1. Format Persentase (contoh: "87%", "87.5 %")
    if '%' in text:
        nums = re.findall(r'\d+(?:\.\d+)?', text)
        if nums:
            return int(float(nums[0]))

    # 2. Format Pecahan (contoh: "8.5/10", "4/5", "8 out of 10")
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)', text)
    if match:
        score = float(match.group(1))
        scale = float(match.group(2))
        if scale > 0:
            return int((score / scale) * 100)

    # 3. Format Angka Murni (contoh: "8.7" atau "87")
    match = re.search(r'^(\d+(?:\.\d+)?)$', text)
    if match:
        score = float(match.group(1))
        if score <= 10:
            return int(score * 10)  # 8.7 -> 87
        elif score <= 100:
            return int(score)       # 87 -> 87

    return 0


class Command(BaseCommand):
    help = "Otomasi penuh untuk menarik data Wikidata via SPARQL per halaman dan merekamnya ke Neo4j."

    def add_arguments(self, parser):
        parser.add_argument(
            '--max-pages',
            type=int,
            default=125,
            help='Batas maksimal halaman yang ingin ditarik (1 halaman = 1000 data)'
        )
        parser.add_argument(
            '--truncate',
            action='store_true',
            default=False,
            help='Set flag ini jika ingin mengosongkan database sebelum proses dimulai'
        )

    def handle(self, *args, **options):
        max_pages = options['max_pages']
        truncate_db = options['truncate']
        limit_per_page = 1000

        wikidata_url = "https://query.wikidata.org/sparql"
        user_agent = "UAS-Neo4j-Collector/1.2"
        headers = {'User-Agent': user_agent, 'Accept': 'application/sparql-results+json'}

        self.stdout.write(self.style.WARNING("🚀 Memulai otomasi sinkronisasi data dari Wikidata ke Neo4j..."))
        self.stdout.write(self.style.WARNING(f"📊 Target maksimal penarikan: {max_pages} halaman (~{max_pages * limit_per_page} baris data)."))
        self.stdout.write("----------------------------------------------------------------")

        # Jalankan perintah pembersihan data di awal jika flag --truncate aktif
        if truncate_db:
            self.stdout.write(self.style.ERROR("⚠️ Menghubungkan ke Neo4j untuk mengosongkan database..."))
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            self.stdout.write(self.style.SUCCESS("✅ Database Neo4j berhasil dikosongkan."))
            print("-" * 40)

        # Perulangan halaman (Otomasi dari fetch_worker.py)
        for page in range(1, max_pages + 1):
            offset = (page - 1) * limit_per_page
            self.stdout.write(self.style.WARNING(f"🔄 Menembak Halaman {page} (Offset: {offset})..."))

            # Query SPARQL dengan teknik Sub-Query Pagination (Dari main.py FastAPI)
            sparql_query = f"""
            SELECT ?filmLabel ?filmDescription ?filmScore ?directorLabel ?directorDescription ?genreLabel ?genreDescription
            WHERE {{
              {{
                SELECT DISTINCT ?film ?director ?genre ?filmScore WHERE {{
                  ?film wdt:P31 wd:Q11424 .
                  ?film wdt:P57 ?director .
                  ?film wdt:P136 ?genre .
                  OPTIONAL {{ ?film wdt:P444 ?filmScore . }}
                }}
                LIMIT {limit_per_page} OFFSET {offset}
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """

            try:
                # Proses Fetching ke Wikidata
                resp = requests.get(wikidata_url, params={'query': sparql_query, 'format': 'json'}, headers=headers, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                bindings = data.get('results', {}).get('bindings', [])

                # Jika data di server Wikidata sudah habis, hentikan perulangan (Otomasi dari fetch_worker.py)
                if not bindings:
                    self.stdout.write(self.style.ERROR(f"🛑 Info: Data di server Wikidata sudah habis pada halaman {page}. Perulangan dihentikan."))
                    break

                # Proses Transformasi & Pembersihan Data (Dari main.py FastAPI)
                records = []
                for item in bindings:
                    f_label = item.get('filmLabel', {}).get('value', 'Unknown').strip()
                    d_label = item.get('directorLabel', {}).get('value', 'Unknown').strip()
                    g_label = item.get('genreLabel', {}).get('value', 'Unknown').strip()

                    # Filter Pengaman: Jangan masukkan jika teks masih berwujud ID Kode (Qxxxxx)
                    if (f_label.startswith("Q") and f_label[1:].isdigit()) or \
                       (d_label.startswith("Q") and d_label[1:].isdigit()) or \
                       (g_label.startswith("Q") and g_label[1:].isdigit()):
                        continue

                    records.append({
                        "filmLabel": f_label,
                        "filmDesc": item.get('filmDescription', {}).get('value', 'No description available').strip(),
                        "filmScore": normalize_to_100(item.get('filmScore', {}).get('value', '')),
                        "directorLabel": d_label,
                        "directorDesc": item.get('directorDescription', {}).get('value', 'No description available').strip(),
                        "genreLabel": g_label,
                        "genreDesc": item.get('genreDescription', {}).get('value', 'No description available').strip()
                    })

                if not records:
                    self.stdout.write(self.style.WARNING(f"ℹ️ Halaman {page} dilewati karena semua baris berisi kode Q-ID."))
                    continue

                # Kueri Ingestion Grafik (Dari main.py FastAPI yang sudah diperbaiki variabel row-nya)
                insert_query = """
                UNWIND $batch AS row
                MERGE (f:Film {title: row.filmLabel})
                  SET f.description = row.filmDesc,
                      f.score = row.filmScore
                MERGE (d:Director {name: row.directorLabel})
                  SET d.description = row.directorDesc
                MERGE (g:Genre {name: row.genreLabel})
                  SET g.description = row.genreDesc
                  
                MERGE (d)-[:DIRECTED]->(f)
                MERGE (f)-[:HAS_GENRE]->(g)
                """

                # Eksekusi Tembakan Batch ke Neo4j
                with driver.session() as session:
                    session.run(insert_query, batch=records)

                self.stdout.write(self.style.SUCCESS(f"✅ Sukses! Halaman {page} berhasil merekam {len(records)} data ke Neo4j."))

            except requests.exceptions.RequestException as req_err:
                self.stdout.write(self.style.ERROR(f"❌ Gagal koneksi HTTP ke Wikidata pada halaman {page}: {str(req_err)}"))
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Terjadi kesalahan sistem pada halaman {page}: {str(e)}"))
                break

            # Jeda waktu 2 detik antar halaman (Otomasi dari fetch_worker.py)
            time.sleep(2)
            print("-" * 40)

        self.stdout.write(self.style.SUCCESS("\n🎉 Proses selesai! Seluruh data hasil penyatuan sistem kini sukses tersimpan di Neo4j."))