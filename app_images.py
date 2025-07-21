import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image
import pytesseract
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Document & Image Q&A", layout="wide")
st.title("📄🖼️ Document & Image Q&A with Streamlit")

# Session state to store documents and embeddings
if "docs" not in st.session_state:
    st.session_state.docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

def split_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text_from_image(image_file):
    image = Image.open(image_file).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text

st.sidebar.header("Upload your documents or images")
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt or image files", 
    type=["txt", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.docs = []
    st.session_state.chunks = []
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if file_type.startswith("text"):
            text = uploaded_file.read().decode("utf-8")
            source = f"Text file: {uploaded_file.name}"
        elif file_type.startswith("image"):
            text = extract_text_from_image(uploaded_file)
            source = f"Image file: {uploaded_file.name}"
            st.image(uploaded_file, caption=source)
        else:
            continue
        st.session_state.docs.append(f"{source}\n\n{text}")
        st.session_state.chunks.extend(split_text(text))
    # Embed all chunks using TF-IDF
    if st.session_state.chunks:
        vectorizer = TfidfVectorizer().fit(st.session_state.chunks)
        st.session_state.chunk_embeddings = vectorizer.transform(st.session_state.chunks)
        st.success(f"Uploaded and processed {len(uploaded_files)} file(s).")
    else:
        st.warning("No valid files uploaded.")

st.header("Ask a question about your documents or images")
question = st.text_input("Your question")
if st.button("Ask") and question:
    if not st.session_state.chunks:
        st.warning("Please upload at least one document or image first.")
    else:
        vectorizer = TfidfVectorizer().fit(st.session_state.chunks + [question])
        question_vec = vectorizer.transform([question])
        chunk_vecs = vectorizer.transform(st.session_state.chunks)
        similarities = cosine_similarity(question_vec, chunk_vecs).flatten()
        top_idx = int(np.argmax(similarities))
        context = st.session_state.chunks[top_idx]
        openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
        if openai_api_key:
            openai.api_key = openai_api_key
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            answer = response.choices[0].message.content.strip()
        else:
            answer = f"(No LLM API key set.) Most relevant context:\n\n{context}"
        st.markdown("**Answer:**")
        st.write(answer)
