# Technical Architecture

## 1. Document Layer

The notebook creates a small text document containing introductory information about LangChain and RAG.

`TextLoader` loads the document into LangChain document objects.

## 2. Chunking Layer

`RecursiveCharacterTextSplitter` divides the document into smaller pieces.

Configuration:

- Chunk size: 100 characters
- Overlap: 20 characters
- Length function: Python `len`
- Separators: paragraph, newline, space, then individual characters

The overlap helps preserve context between neighboring chunks.

## 3. Embedding Layer

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

The embedding model converts each text chunk into a numerical vector.

## 4. Vector Database

FAISS stores the generated vectors and supports fast similarity search.

The notebook creates the index with:

```python
db = FAISS.from_documents(
    documents=splits,
    embedding=embedding_function
)
```

## 5. Retriever

The FAISS database is converted into a LangChain retriever:

```python
retriever = db.as_retriever(search_kwargs={"k":2})
```

The retriever returns the top two relevant chunks.

## 6. Generation Layer

The generation model is:

`Qwen/Qwen2-0.5B-Instruct`

The notebook loads the tokenizer and causal language model through Hugging Face Transformers and wraps the generation pipeline with LangChain's `HuggingFacePipeline`.

## 7. Prompt Formatting

A custom function creates Qwen-compatible chat messages containing:

- System instructions
- Retrieved context
- User question

The system prompt tells Qwen to answer only from the provided context.

## 8. End-to-End Flow

```text
Question
   |
   v
FAISS Retriever
   |
   v
Top 2 Relevant Chunks
   |
   v
Context Formatting
   |
   v
Qwen Chat Template
   |
   v
Qwen2-0.5B-Instruct
   |
   v
Answer
```

## 9. Current Scope

This implementation is intentionally small and educational. It demonstrates the core RAG pattern rather than a production deployment architecture.

The current notebook does not include:

- Persistent FAISS index saving/loading
- Web or API frontend
- Authentication
- Conversation memory
- Evaluation benchmarks
- Production logging
- Document ingestion from multiple formats
