# Menggunakan base image Python 3.12 yang ringan (slim)
FROM python:3.12-slim

# Mengatur environment variable untuk optimasi Python di dalam kontainer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Mengatur folder kerja utama di dalam kontainer
WORKDIR /app

# Menginstal dependensi sistem yang dibutuhkan untuk kompilasi beberapa library Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Menyalin berkas requirements.txt terlebih dahulu untuk memanfaatkan caching Docker
COPY requirements.txt /app/

# Mengunduh dan memasang seluruh library Python pendukung proyek
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh sisa kode sumber proyek ke dalam kontainer
COPY . /app/

# Membuka jalur port 8000 untuk akses aplikasi web Django
EXPOSE 8000

# Perintah default untuk menjalankan server Django saat kontainer aktif
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]