Consumer Rights Assistant – RAG-Based AI System

Overview:
The Consumer Rights Assistant is a Retrieval-Augmented Generation (RAG) based AI system designed to answer consumer rights queries using official PDF documents.
Unlike a standard chatbot, this system retrieves relevant content from trusted consumer rights documents and generates grounded responses using a local Large Language Model (LLM).

Key Features:
-Semantic Similarity Search using Sentence Transformers
-FAISS Vector Database for efficient document retrieval
-Local LLM (Mistral via Ollama) for grounded response generation
-Answers strictly based on official consumer rights PDFs
-Streamlit-based interactive UI

Tech Stack:
-Python
-Streamlit
-Sentence Transformers (all-mpnet-base-v2)
-FAISS (Vector Database)
-Ollama (Local LLM – Mistral)
-LangChain (Document Processing)

Why RAG?
Traditional LLMs may hallucinate.
This system reduces hallucination by grounding responses in verified consumer rights documents.

Built as an AI/NLP engineering project focusing on semantic retrieval and grounded generation.
