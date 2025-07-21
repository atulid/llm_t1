import streamlit as st
import tempfile
import os
import numpy as np
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Document Q&A", layout="wide")

st.title("📄 Document Q&A with Streamlit")

# Session state to store documents and embeddings
if "docs" not in st.session_state:
    st.session_state.docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

# Helper: Split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Upload and process documents
st.sidebar.header("Upload your documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt files", type=["txt"], accept_multiple_files=True
)

if uploaded_files:
    st.session_state.docs = []
    st.session_state.chunks = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.docs.append(text)
        st.session_state.chunks.extend(split_text(text))
    # Embed all chunks using TF-IDF
    vectorizer = TfidfVectorizer().fit(st.session_state.chunks)
    st.session_state.chunk_embeddings = vectorizer.transform(st.session_state.chunks)
    st.success(f"Uploaded and processed {len(uploaded_files)} document(s).")

st.header("Ask a question about your documents")

question = st.text_input("Your question")
if st.button("Ask") and question:
    if not st.session_state.chunks:
        st.warning("Please upload at least one document first.")
    else:
        # Embed question and find most relevant chunk(s)
        vectorizer = TfidfVectorizer().fit(st.session_state.chunks + [question])
        question_vec = vectorizer.transform([question])
        chunk_vecs = vectorizer.transform(st.session_state.chunks)
        similarities = cosine_similarity(question_vec, chunk_vecs).flatten()
        top_idx = int(np.argmax(similarities))
        context = st.session_state.chunks[top_idx]

        # Use OpenAI API for answer generation (optional, requires API key)
        openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
        if openai_api_key:
            openai.api_key = openai_api_key
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            answer = response.choices[0].message.content.strip()
        else:
            # Fallback: simple context display
            answer = f"(No LLM API key set.) Most relevant context:\n\n{context}"

        st.markdown("**Answer:**")
        st.write(answer)
