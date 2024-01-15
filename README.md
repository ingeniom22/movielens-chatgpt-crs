## Movielens Conversational Recommender System with ChatGPT and DeepFM

## Gunakan Python versi 3.11.6

### Tahap Instalasi

#### 1. Membuat dan Mengaktifkan Lingkungan Virtual (venv)
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Library dari requirements.txt
```bash
pip install -r requirements.txt
```

### Persiapan File .env

#### 3. Membuat dan Mengisi File .env, gunakan file .env.example sebagai contoh
Buat file `.env` di direktori proyek dan isi dengan informasi berikut:
```ini
# .env
OPENAI_API_KEY=isi_api_key_anda_di_sini
```

### Memilih Version
#### 3. Checkout Versi App
melalui terminal VSCode, jalankan command:
```bash
git checkout v1.1 # untuk pindah ke version 1.1
git pull
```


### Menjalankan Aplikasi

#### 4. Jalankan main.py
```bash
python main_lkpp.py # atau python main_movielens.py
```
