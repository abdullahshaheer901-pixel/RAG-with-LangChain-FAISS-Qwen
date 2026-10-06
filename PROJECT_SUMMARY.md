# Project Summary

## Title

RAG with LangChain, FAISS & Qwen

## Objective

Build a simple Retrieval-Augmented Generation pipeline that retrieves relevant information from a local document and uses a lightweight Qwen instruction model to generate an answer from that retrieved context.

## Workflow

Document → Chunking → Embeddings → FAISS → Retrieval → Context + Question → Qwen → Answer

## Key Components

- `TextLoader` for document loading
- `RecursiveCharacterTextSplitter` for chunking
- `all-MiniLM-L6-v2` for embeddings
- `FAISS` for vector similarity search
- `Qwen2-0.5B-Instruct` for generation
- LangChain runnables for pipeline composition

## Learning Outcomes

This project demonstrates the basic concepts behind:

- Semantic embeddings
- Vector databases
- Similarity retrieval
- Prompt construction
- Retrieval-Augmented Generation
- Local Hugging Face model inference
- LangChain pipeline composition
