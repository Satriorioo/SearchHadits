import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# === CONFIG: ID Google Drive file (ubah ini sesuai file kamu) ===
GDRIVE_EMBED_ID = "1UN0_TErNx3azqniMQYxwgrS9L3uudkFB"  # <-- ID hadits_embeddings.npy
GDRIVE_META_ID = "1EnhP8h60Bzu-CYRXnJ_gMZhTR2GBeCMX"   # <-- ID hadits_meta.pkl

# === Download file dari Google Drive jika belum ada ===
def download_if_not_exists():
    if not os.path.exists("hadits_embeddings.npy"):
        st.info("Mengunduh file embedding model... (hanya sekali)")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_EMBED_ID}", "hadits_embeddings.npy", quiet=False)
    if not os.path.exists("hadits_meta.pkl"):
        st.info("Mengunduh file metadata hadits... (hanya sekali)")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_META_ID}", "hadits_meta.pkl", quiet=False)

download_if_not_exists()

# === Load model dan data (dengan caching agar cepat) ===
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_data():
    df = pd.read_pickle("hadits_meta.pkl")
    embeddings = np.load("hadits_embeddings.npy")
    return df, embeddings

model = load_model()
df, embeddings = load_data()

# Fungsi pencarian dipisah menjadi tiga bagian ===

def keyword_search(query, top_n=5):
    """Fungsi untuk pencarian berbasis kata kunci (literal)."""
    # Mencari query sebagai substring di dalam kolom 'Terjemahan', tidak case-sensitive
    mask = df['Terjemahan'].str.contains(query, case=False, na=False)
    results = df[mask].head(top_n).copy()
    # Menambahkan kolom untuk menandai metode pencarian
    if not results.empty:
        results["search_type"] = "Keyword"
    return results

def semantic_search(query, top_n=5):
    """Fungsi untuk pencarian berbasis kemiripan makna (semantik)."""
    query_vec = model.encode([query])
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argsort(sim_scores)[::-1][:top_n]
    results = df.iloc[top_idx].copy()
    results["similarity"] = sim_scores[top_idx]
    # Menambahkan kolom untuk menandai metode pencarian
    results["search_type"] = "Semantik"
    return results

def hybrid_search(query, top_n=5):
    """Gabungkan hasil pencarian keyword dan semantik, hindari duplikat."""
    keyword_results = keyword_search(query, top_n)
    semantic_results = semantic_search(query, top_n)

    # Buang duplikat berdasarkan isi 'Terjemahan'
    combined = pd.concat([keyword_results, semantic_results])
    combined = combined.drop_duplicates(subset=["Terjemahan"]).reset_index(drop=True)

    return combined


# === UI Streamlit ===
st.title("🔍 Pencarian Hadits (Keyword & Semantik)")
st.markdown("Masukkan kata atau kalimat, misalnya: *mabuk*, *pacaran*, *menjaga pandangan*")

query = st.text_input("Masukkan kata/kalimat:")
top_n = st.slider("Jumlah hasil ditampilkan:", 1, 10, 5)

if query:
    # Menggunakan fungsi hybrid_search yang baru
    results = hybrid_search(query, top_n)
    
    st.markdown("### ✨ Hasil Pencarian:")
    if results.empty:
        st.warning("Tidak ada hadis yang ditemukan untuk kueri Anda.")
    else:
        for i, row in results.iterrows():
            # Menampilkan informasi metode pencarian yang digunakan
            search_method_display = f"**Metode Pencarian**: `{row['search_type']}`"

            # Tampilkan skor kemiripan sebagai persen jika metodenya semantik
            similarity_display = ""
            if row['search_type'] == 'Semantik':
                similarity_display = f"**📊 Skor Kemiripan**: `{row['similarity'] * 100:.2f}%`"

            st.markdown(f"""
            {search_method_display}

            **🧾 Perawi**: {row['Perawi']}

            **📖 Hadits**: {row['Terjemahan']}

            {similarity_display}
            ---
            """)
