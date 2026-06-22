import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Memuat konfigurasi dari file .env
load_dotenv()

print("DEBUG URI:", os.getenv("NEO4J_URI"))
print("DEBUG USER:", os.getenv("NEO4J_USER"))

# Inisialisasi driver koneksi ke Neo4j secara global
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def search_movie_hybrid(query_vector, limit=3):
    """
    Fungsi untuk mencari node Film terdekat menggunakan Vector Index,
    lalu menelusuri (traverse) relasi ke Director dan Genre.
    """
    # Pastikan nama indeks 'movie_vector_index' sesuai dengan yang Anda buat di Neo4j
    query = """
    CALL db.index.vector.queryNodes('movie_vector_index', $limit, $vector)
    YIELD node AS film, score
    MATCH (d:Director)-[:DIRECTED]->(film)
    MATCH (film)-[:HAS_GENRE]->(g:Genre)
    RETURN film.title AS title, 
           film.description AS description, 
           film.score AS rating, 
           d.name AS director, 
           collect(g.name) AS genres
    """
    
    context_list = []
    
    with driver.session() as session:
        result = session.run(query, vector=query_vector, limit=limit)
        for record in result:
            genres_str = ", ".join(record["genres"])
            # Format data graf menjadi potongan teks naratif
            movie_info = (
                f"Film: {record['title']}\n"
                f"Rating/Score: {record['rating']}\n"
                f"Sutradara: {record['director']}\n"
                f"Genre: {genres_str}\n"
                f"Sinopsis: {record['description']}\n"
                f"---"
            )
            context_list.append(movie_info)
            
    # Gabungkan semua data film yang relevan menjadi satu string utuh
    return "\n\n".join(context_list) if context_list else "Tidak ada data film yang relevan di dalam database."