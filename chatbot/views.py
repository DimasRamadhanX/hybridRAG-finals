from django.shortcuts import render
from django.http import HttpResponse
from .services.rag_service import generate_multi_hop_answer
import re

def chat_view(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "").strip()
        
        # Jaring pengaman jika user mengirim pesan kosong
        if not user_message:
            return HttpResponse("")

        # Inisialisasi chat history di dalam session jika belum ada
        if 'chat_history' not in request.session:
            request.session['chat_history'] = []
        history = request.session['chat_history']

        try:
            # Mengambil maksimal 4 percakapan terakhir sebagai konteks ingatan LLM
            recent_history = history[-4:] if history else []
            
            # MENANGKAP TUPLE: Jawaban final bot dan daftar log jalur berpikirnya
            bot_response, reasoning_logs = generate_multi_hop_answer(user_message, history=recent_history)
            
        except Exception as e:
            bot_response = f"Maaf, terjadi kendala teknis pada sistem: {str(e)}"
            reasoning_logs = ["❌ Sistem mengalami kegagalan eksekusi internal."]
        
        # Simpan pesan terbaru ke dalam riwayat session
        history.append({"user": user_message, "bot": bot_response})
        request.session['chat_history'] = history
        request.session.modified = True
        
        # SINKRONISASI FRAGMEN DI VIEWS.PY DENGAN CSS TERBARU CHAT.HTML
        # Membangun elemen list langkah berpikir dari array log
        steps_html = "".join([
            f'<li class="reasoning-step">{step}</li>'
            for step in reasoning_logs
        ])

        # Merakit fragmen HTML respons untuk disisipkan langsung oleh HTMX
        html_fragment = f"""
        <div class="chat-message user">
            <div class="bubble">{user_message}</div>
        </div>

        <div class="chat-message bot">
            <div class="bubble">
                <details class="reasoning-box">
                    <summary>🔮 Lihat Jejak Berpikir Agen (Reasoning Path) <span>▼</span></summary>
                    <div class="reasoning-content">
                        <ul class="reasoning-list">
                            {steps_html}
                        </ul>
                    </div>
                </details>

                <p class="bot-title">🎬 Movie Assistant</p>
                <p class="bot-text">{bot_response}</p>
            </div>
        </div>
        """
        return HttpResponse(html_fragment)

    # Jalur GET: Menampilkan halaman utama chat pertama kali
    return render(request, "chatbot/chat.html")

def run_analytics_view(request):
    """
    Endpoint POST untuk menjalankan proses Graph Analytics (PageRank) secara dinamis
    dan mengembalikan fragmen HTML berisi log proses serta daftar entitas terpopuler.
    """
    if request.method != "POST":
        return HttpResponse("Hanya mendukung metode POST", status=405)
        
    from .services.neo4j_client import driver
    
    q_drop = "CALL gds.graph.drop('katalog-film-projeksi', false)"
    q_project = "CALL gds.graph.project('katalog-film-projeksi', ['Director', 'Film'], 'DIRECTED')"
    q_pagerank_write = "CALL gds.pageRank.write('katalog-film-projeksi', {writeProperty: 'pagerank_score'})"
    q_pagerank_stream = """
        CALL gds.pageRank.stream('katalog-film-projeksi')
        YIELD nodeId, score
        RETURN 
            coalesce(gds.util.asNode(nodeId).title, gds.util.asNode(nodeId).name) AS nama_entitas,
            labels(gds.util.asNode(nodeId))[0] AS tipe_node,
            score
        ORDER BY score DESC LIMIT 5
    """
    
    logs = []
    results = []
    error_msg = None
    
    try:
        with driver.session() as session:
            # 1. Drop old projection
            session.run(q_drop)
            logs.append("🗑️ Projeksi memori graf lama dibersihkan.")
            
            # 2. Project new graph
            session.run(q_project)
            logs.append("🏗️ Projeksi graf baru 'katalog-film-projeksi' dibuat.")
            
            # 3. Write PageRank score
            session.run(q_pagerank_write)
            logs.append("💾 Skor PageRank ditulis ke properti 'pagerank_score' pada database.")
            
            # 4. Stream top 5 entities
            res = session.run(q_pagerank_stream)
            for record in res:
                results.append({
                    "name": record["nama_entitas"] or "Tanpa Nama",
                    "type": record["tipe_node"] or "Unknown",
                    "score": record["score"] or 0.0
                })
            logs.append("🏆 Hasil PageRank teratas berhasil dieksekusi.")
            
    except Exception as e:
        error_msg = f"Gagal mengeksekusi GDS PageRank: {str(e)}"
        logs.append(f"❌ Terjadi kesalahan pada proses analisis.")
        
    # Membangun HTML respon
    logs_html = "".join([f'<li class="log-item">{log}</li>' for log in logs])
    
    if error_msg:
        content_html = f"""
        <div class="analytics-error">
            <p>{error_msg}</p>
        </div>
        """
    else:
        rows_html = ""
        for i, item in enumerate(results, 1):
            badge_class = "badge-film" if item["type"] == "Film" else "badge-director"
            type_label = "Film" if item["type"] == "Film" else "Sutradara"
            # Format score to display beautifully as influence weight percentage
            score_percent = f"{round(item['score'] * 100)}%"
            rows_html += f"""
            <tr class="analytics-row">
                <td class="col-rank">{i}</td>
                <td class="col-name">{item['name']}</td>
                <td class="col-type"><span class="badge {badge_class}">{type_label}</span></td>
                <td class="col-score">{score_percent}</td>
            </tr>
            """
            
        content_html = f"""
        <div class="analytics-success">
            <table class="analytics-table">
                <thead>
                    <tr>
                        <th class="col-rank">#</th>
                        <th class="col-name">Entitas</th>
                        <th class="col-type">Tipe</th>
                        <th class="col-score">Pengaruh</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        
    html_fragment = f"""
    <div class="analytics-results-container">
        <div class="analytics-logs-box">
            <p class="logs-title">📋 Log Eksekusi GDS:</p>
            <ul class="logs-list">
                {logs_html}
            </ul>
        </div>
        {content_html}
    </div>
    """
    return HttpResponse(html_fragment)