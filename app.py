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
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_EMBED_ID}", "hadits_embeddings.npy", quiet=False)
    if not os.path.exists("hadits_meta.pkl"):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_META_ID}", "hadits_meta.pkl", quiet=False)

download_if_not_exists()

# === Load model dan data ===
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

# === Fungsi pencarian hadits ===
def search_hadits(query, top_n=5):
    query_vec = model.encode([query])
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argsort(sim_scores)[::-1][:top_n]
    results = df.iloc[top_idx].copy()
    results["similarity"] = sim_scores[top_idx]
    return results

# === UI Streamlit ===
st.title("🔍 Pencarian Hadits Berdasarkan Makna (Semantic Search)")
st.markdown("Masukkan kata atau kalimat, misalnya: *pacaran*, *zina*, *menjaga pandangan*")

query = st.text_input("Masukkan kata/kalimat:")
top_n = st.slider("Jumlah hasil ditampilkan:", 1, 10, 5)

if query:
    results = search_hadits(query, top_n)
    st.markdown("### ✨ Hasil Pencarian:")
    for i, row in results.iterrows():
        st.markdown(f"""
        **🧾 Perawi**: {row['Perawi']}

        **📖 Hadits**: {row['Terjemahan']}
          
        **📊 Skor Kemiripan**: `{row['similarity']:.4f}`  
        ---
        """)
