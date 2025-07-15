
import streamlit as st
import json
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Sanitation Chatbot", page_icon="🧼")
st.title("💬 Sanitation & Hygiene Chatbot")

# Debug checkpoint
st.write("✅ App started loading")

# Load all_chunks.json
try:
    with open("all_chunks.json", "r") as f:
        all_chunks = json.load(f)
    st.write("✅ all_chunks.json loaded")
except Exception as e:
    st.error(f"❌ Failed to load all_chunks.json: {e}")
    st.stop()

# Load FAISS index
try:
    index = faiss.read_index("faiss_index/index.faiss")
    st.write("✅ FAISS index loaded")
except Exception as e:
    st.error(f"❌ Failed to load faiss_index/index.faiss: {e}")
    st.stop()

# Load sentence transformer model
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.write("✅ Embedding model loaded")
except Exception as e:
    st.error(f"❌ Failed to load embedding model: {e}")
    st.stop()

# Load lightweight text generation model (safe for Streamlit Cloud)
try:
    qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device=-1)
    st.write("✅ LLM loaded (Falcon-RW-1B, CPU)")
except Exception as e:
    st.error(f"❌ Failed to load language model: {e}")
    st.stop()

# Function: search relevant chunks
def search_similar_chunks(query, top_k=5):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)
    matched = [all_chunks[i]["text"] for i in indices[0] if i < len(all_chunks)]
    return "\n\n".join(matched)

# Function: ask chatbot
def ask_chatbot(question):
    try:
        context = search_similar_chunks(question)
        if not context.strip():
            return "I don't know. The answer is not available in the provided knowledge base."

        prompt = f"""Answer the question only using the context below. If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
Answer:"""

        response = qa_pipeline(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
        return response.split("Answer:")[-1].strip()

    except Exception as e:
        return f"⚠️ Internal error: {e}"

# UI interaction
user_input = st.text_input("Ask your sanitation-related question:")

if user_input:
    with st.spinner("🤖 Thinking..."):
        answer = ask_chatbot(user_input)
        st.markdown(f"**Bot:** {answer}")
        
