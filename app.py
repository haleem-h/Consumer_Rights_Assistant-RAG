import streamlit as st
import ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(page_title="Consumer Rights Assistant", page_icon="🤖",layout="centered")


st.markdown("""<style>body { background-color: #000000; }.stApp { background-color: #000000; color: white; }.chat-bubble { padding: 14px; border-radius: 12px; margin-bottom: 12px; }.user { background: linear-gradient(145deg, #2c2c2c, #1a1a1a); color: white; }
.bot { background: linear-gradient(145deg, #cfd0d0, #8a8b8d); color: black; }input { background-color: #111 !important; color: white !important; }</style>""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🤖 Consumer Rights Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Hello! How can I assist you today?</p>", unsafe_allow_html=True)


@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local("vector_db",embeddings,allow_dangerous_deserialization=True)

vectorstore = load_vector_db()

def clean_text(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())


query = st.text_input("Type here...", key="query")

if query:
    st.markdown(f"<div class='chat-bubble user'>🧑 {query}</div>",unsafe_allow_html=True)  
    docs = vectorstore.similarity_search(query, k=8)

    if not docs:
        st.markdown("<div class='chat-bubble bot'>🤖 Information not found in documents.</div>", unsafe_allow_html=True)
    else:
        context = "\n\n".join([clean_text(d.page_content) for d in docs])

        prompt = f"""You are a Consumer Rights Assistant in India.
        Answer the question using ONLY the context below.
        Explain clearly in simple step-by-step instructions.
        Do not mention document names.
        If answer not found, say: Information not found in documents.

        Question:{query}
        Context:{context}
        Answer:"""

        response = ollama.chat(model="mistral",messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

        st.markdown(f"<div class='chat-bubble bot'>🤖 {answer}</div>",unsafe_allow_html=True)
