#  RAG with LangChain, FAISS & Qwen

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Database-orange?style=flat-square)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface&logoColor=white)
![Qwen](https://img.shields.io/badge/Qwen-2.0--0.5B-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

A hands-on **Retrieval-Augmented Generation (RAG)** project built with **LangChain, FAISS, Hugging Face Sentence Transformers, and Qwen2-0.5B-Instruct**.

The project demonstrates how a language model can retrieve relevant information from a document and generate an answer grounded in the retrieved context.

---

## Table of Contents

- [Overview](#overview)
- [RAG Architecture](#rag-architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Learning Path](#learning-path)
- [Results](#results)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository focuses on understanding and implementing a basic **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of asking the language model to answer only from its internal knowledge, the system first retrieves relevant information from a local document and then provides that information as context to the language model.

The project includes:

- Document loading with LangChain
- Recursive text splitting
- Text embeddings using `all-MiniLM-L6-v2`
- FAISS vector database
- Similarity-based document retrieval
- Qwen2-0.5B-Instruct language model
- Custom prompt formatting for grounded answers
- LangChain runnable-based RAG pipeline
- Example question-answering workflow

---

## RAG Architecture

The complete workflow can be represented as:

```text
                    ┌──────────────────┐
                    │   Text Document  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Text Loader    │
                    └────────┬─────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │ Recursive Text Splitter    │
              │ Chunk Size: 100            │
              │ Overlap: 20                │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Sentence Transformer      │
              │ all-MiniLM-L6-v2           │
              └─────────────┬──────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  FAISS Vector   │
                    │    Database     │
                    └────────┬────────┘
                             │
                        Top K = 2
                             │
                             ▼
                    ┌─────────────────┐
                    │ Retrieved       │
                    │ Context         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Qwen2-0.5B      │
                    │   Instruct      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Grounded Answer │
                    └─────────────────┘
```

---

## Key Features

- 🔹 Retrieval-Augmented Generation pipeline
- 🔹 LangChain document processing
- 🔹 Recursive text chunking
- 🔹 Semantic embeddings with Sentence Transformers
- 🔹 FAISS similarity search
- 🔹 Top-2 relevant document retrieval
- 🔹 Qwen2-0.5B-Instruct integration
- 🔹 Context-grounded answer generation
- 🔹 Custom prompt formatting
- 🔹 Local document-based question answering
- 🔹 Jupyter Notebook implementation
- 🔹 No external API key required for the demonstrated local workflow

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **LangChain** | RAG pipeline orchestration |
| **FAISS** | Vector similarity search |
| **Hugging Face Transformers** | LLM and tokenizer |
| **Sentence Transformers** | Text embeddings |
| **Qwen2-0.5B-Instruct** | Answer generation |
| **PyTorch** | Deep Learning backend |
| **Jupyter Notebook** | Development and experimentation |

### Models Used

**Embedding Model**

```text
sentence-transformers/all-MiniLM-L6-v2
```

**Language Model**

```text
Qwen/Qwen2-0.5B-Instruct
```

---

## Project Structure

```text
RAG-with-LangChain-FAISS-Qwen/
│
├── 📓 RAG_PipeLines.ipynb
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 ARCHITECTURE.md
├── 📄 SETUP.md
├── 📄 PROJECT_SUMMARY.md
├── 📄 PROJECT_REPORT.docx
├── 📄 LICENSE
└── 🚫 .gitignore
```

The notebook contains the complete demonstrated RAG workflow, from embeddings and document indexing through retrieval and Qwen-based generation.

---

## Getting Started

### Prerequisites

Make sure you have:

- Python 3.8+
- pip
- Jupyter Notebook / JupyterLab
- Git
- Sufficient RAM/storage for the Hugging Face models
- GPU is optional; `device_map="auto"` is used in the notebook

### 1. Clone the Repository

```bash
git clone https://github.com/abdullahshaheer901-pixel/RAG-with-LangChain-FAISS-Qwen.git

cd RAG-with-LangChain-FAISS-Qwen
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
RAG_PipeLines.ipynb
```

Run the notebook cells sequentially.

---

## Usage

The notebook demonstrates the following workflow.

### 1. Create Embeddings

The project uses:

```python
sentence-transformers/all-MiniLM-L6-v2
```

to convert text into numerical vector representations.

### 2. Load and Split Documents

A text document is loaded using LangChain's `TextLoader`.

The project uses:

```text
Chunk Size: 100
Chunk Overlap: 20
```

with `RecursiveCharacterTextSplitter`.

### 3. Build the FAISS Index

The generated embeddings are stored in FAISS:

```python
FAISS.from_documents(...)
```

This enables similarity-based retrieval.

### 4. Configure the Retriever

The FAISS database is converted into a retriever:

```python
db.as_retriever(search_kwargs={"k": 2})
```

The system retrieves the two most relevant document chunks.

### 5. Load Qwen

The project uses:

```text
Qwen/Qwen2-0.5B-Instruct
```

through Hugging Face Transformers and LangChain's `HuggingFacePipeline`.

### 6. Generate an Answer

The retrieved context and user question are passed through the RAG chain.

Example:

```text
What dose RAG Stand for?
```

The model then generates an answer using the retrieved context.

---

## How the Pipeline Works

```text
1. Load Document
       ↓
2. Split Document into Chunks
       ↓
3. Generate Embeddings
       ↓
4. Store Embeddings in FAISS
       ↓
5. User Asks a Question
       ↓
6. Retrieve Top 2 Relevant Chunks
       ↓
7. Build Context
       ↓
8. Send Context + Question to Qwen
       ↓
9. Generate Grounded Answer
```

---

## Grounded Generation Strategy

The project uses a custom prompt format designed to keep the generated answer grounded in the retrieved context.

The model is instructed to:

- Answer using the provided context.
- Avoid relying on external knowledge.
- Avoid unsupported information.
- Return:

```text
I don't know based on the provided context.
```

when the requested information is not available in the supplied context.

Generated responses are also formatted to begin with:

```text
Answer:
```

This makes the output structure consistent.

---

## Learning Path

This project can be used as part of a practical **Generative AI / RAG learning path**:

```text
Python
   ↓
NLP Fundamentals
   ↓
Embeddings
   ↓
Vector Databases
   ↓
Semantic Search
   ↓
LangChain
   ↓
Retrieval-Augmented Generation
   ↓
LLMs
   ↓
Grounded Question Answering
   ↓
Production RAG Applications
```

---

## Results

The project successfully demonstrates the core components of a local RAG workflow:

- Document ingestion
- Text chunking
- Embedding generation
- Vector indexing with FAISS
- Similarity-based retrieval
- Context construction
- Qwen-based answer generation

The demonstrated notebook uses a small sample text document and an example question to show the complete pipeline.

---

## Roadmap

- [x] Implement document loading
- [x] Implement text splitting
- [x] Generate text embeddings
- [x] Create FAISS vector index
- [x] Configure similarity retriever
- [x] Integrate Qwen2-0.5B-Instruct
- [x] Build LangChain RAG chain
- [x] Add grounded prompting
- [ ] Add PDF document ingestion
- [ ] Add DOCX document ingestion
- [ ] Support multiple documents
- [ ] Add persistent FAISS storage
- [ ] Add source citations
- [ ] Add retrieval evaluation
- [ ] Build Streamlit interface
- [ ] Build FastAPI backend
- [ ] Add conversation memory
- [ ] Deploy the RAG application

---

## Contributing

Contributions, suggestions, issues, and feature requests are welcome.

### Contribution Steps

```bash
# Fork the repository

# Create a feature branch
git checkout -b feature/your-feature

# Commit your changes
git commit -m "Add your feature"

# Push the branch
git push origin feature/your-feature
```

Then open a Pull Request.

---

## License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

## Author

**Abdullah Shaheer**

Data Scientist | Data Analyst | Machine Learning & Deep Learning | Computer Vision | GenAI | LangChain & RAG

### Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullah-shaheer260)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/abdullahshaheer260)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/abdullahshaheer901-pixel)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:abdullahshaheer901@gmail.com)

---

## Acknowledgements

This project uses the following open-source technologies:

- [LangChain](https://www.langchain.com/) — RAG and LLM application framework
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Hugging Face](https://huggingface.co/) — Models and Transformers
- [Qwen](https://huggingface.co/Qwen) — Instruction-tuned language model
- [Sentence Transformers](https://www.sbert.net/) — Text embeddings
- [PyTorch](https://pytorch.org/) — Machine learning framework

---

## ⭐ Support

If this project helped you learn about **RAG, LangChain, FAISS, embeddings, vector databases, or LLM applications**, consider giving the repository a ⭐.

**Thank you for visiting! 🚀**
