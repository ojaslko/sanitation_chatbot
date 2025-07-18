
import json
import faiss
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import gradio as gr

# Load FAISS index
index = faiss.read_index("faiss_index/index.faiss")

# Load your data chunks
with open("all_chunks.json", "r") as f:
    chunks = json.load(f)

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Zephyr model (local or Hugging Face API)
model_name = "HuggingFaceH4/zephyr-7b-alpha"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Search similar context
def search_similar_chunks(query, k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype("float32"), k)
    return "

".join([chunks[i] for i in indices[0]])

# Ask function
def ask_chatbot(question):
    context = search_similar_chunks(question)
    prompt = f"""Answer the question only using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    response = qa_pipeline(prompt, max_new_tokens=300, do_sample=True, temperature=0.3)[0]["generated_text"]
    return response.split("Answer:")[-1].strip()

# Gradio UI
demo = gr.Interface(fn=ask_chatbot, inputs="text", outputs="text", title="SANTBOT: Sanitation Chatbot")

# Start server (Render will run this line)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
