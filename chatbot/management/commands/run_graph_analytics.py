import os
import re
from django.core.management.base import BaseCommand
from neo4j import GraphDatabase

class Command(BaseCommand):
    help = "Menjalankan analisis centrality (PageRank) menggunakan Neo4j GDS Plugin via Cypher untuk syarat Tier 1"

    def handle(self, *args, **options):
        # 1. Ambil kredensial database dari environment variable
        # Menyesuaikan dengan konfigurasi URI dan USER yang muncul di log debug kamu
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password") # Ganti dengan password asli di .env jika berbeda

        self.stdout.write(self.style.MIGRATE_HEADING("📊 Memulai Proses Graph Analytics (Neo4j GDS)..."))

        # 2. Definisikan rangkaian query Cypher GDS
        q_drop = "CALL gds.graph.drop('katalog-film-projeksi', false)"
        q_project = "CALL gds.graph.project('katalog-film-projeksi', ['Director', 'Film'], 'DIRECTED')"
        q_pagerank = """
            CALL gds.pageRank.stream('katalog-film-projeksi')
            YIELD nodeId, score
            RETURN 
                coalesce(gds.util.asNode(nodeId).title, gds.util.asNode(nodeId).name) AS nama_entitas,
                labels(gds.util.asNode(nodeId))[0] AS tipe_node,
                score
            ORDER BY score DESC LIMIT 5
        """

        try:
            # 3. Inisialisasi driver Neo4j
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Tahap A: Bersihkan projeksi memori lama jika ada
                session.run(q_drop)
                self.stdout.write(self.style.SUCCESS("🗑️  Projeksi graf lama berhasil dibersihkan."))
                
                # Tahap B: Buat projeksi graf baru ke dalam memori GDS
                session.run(q_project)
                self.stdout.write(self.style.SUCCESS("🏗️  Projeksi graf baru 'katalog-film-projeksi' berhasil dibuat."))
                
                # Tahap C: Eksekusi streaming algoritma PageRank
                self.stdout.write("\n" + "=" * 65)
                self.stdout.write(self.style.WARNING("🏆 HASIL ANALISIS CENTRALITY (PAGERANK) TERTINGGI:"))
                self.stdout.write("=" * 65)
                self.stdout.write(f"{'Nama Entitas':<35} | {'Tipe Node':<12} | {'Skor PageRank'}")
                self.stdout.write("-" * 65)
                
                results = session.run(q_pagerank)
                for record in results:
                    nama = record['nama_entitas'] or "Tanpa Nama"
                    tipe = record['tipe_node'] or "Unknown"
                    skor = record['score'] or 0.0
                    self.stdout.write(f"{nama:<35} | {tipe:<12} | {skor:.4f}")
                    
                self.stdout.write("=" * 65 + "\n")
                self.stdout.write(self.style.SUCCESS("🎉 Fitur Graph Analytics GDS selesai dieksekusi dengan sukses!"))

            driver.close()

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Terjadi kesalahan saat mengeksekusi GDS: {str(e)}"))