# AI-Learning3: Q&A Assistant with RAG and Azure AI

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Hugging Face embeddings, ChromaDB, and Azure OpenAI. It allows users to ask natural language questions about a PDF documentation, and receive answers based on its content.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `requirements.txt` | Lists all dependencies required to run the project |
| `rag_runner_chromadb.py` | Main script that loads the resume PDF, chunks it, stores embeddings in a vector DB, and uses Azure OpenAI to answer user queries |
| `rag_runner_supabase.py` | Main script that loads a PDF about Intouch product, chunks it, stores embeddings in Supabase vector DB, and uses Azure OpenAI to answer user queries |

---

## 🚀 Features

- ✅ Loads and parses a document in PDF format
- ✅ Chunks the text using `RecursiveCharacterTextSplitter`
- ✅ Embeds chunks using Hugging Face’s `all-MiniLM-L6-v2` model
- ✅ Stores embeddings in a local ChromaDB (or Supabase DB.)

## 📦 Requirements

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Make sure your .env file contains:
```bash
AZURE_ENDPOINT=your_azure_openai_endpoint
AZURE_API_KEY=your_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```
