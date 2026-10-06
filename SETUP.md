# Setup & Run Guide

## Requirements

Recommended:

- Python 3.9+
- Internet connection for first-time model downloads
- At least several GB of available storage for model/cache files
- CPU is supported; GPU can improve performance

## 1. Clone the repository

```bash
git clone https://github.com/abdullahshaheer901-pixel/RAG-with-LangChain-FAISS-Qwen.git
cd RAG-with-LangChain-FAISS-Qwen
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Jupyter

```bash
jupyter notebook
```

Open `RAG_PipeLines.ipynb`.

## 5. Run cells in order

The notebook follows this sequence:

1. Initialize embeddings.
2. Create sample content.
3. Generate an embedding for the content.
4. Create/load the document.
5. Split the document into chunks.
6. Build the FAISS vector database.
7. Download/load Qwen.
8. Build the Qwen generation pipeline.
9. Configure the retriever.
10. Build the RAG chain.
11. Ask a question and inspect the retrieved context.

## Troubleshooting

### FAISS installation issue

Try:

```bash
pip install faiss-cpu
```

### Torch/model loading issue

Make sure PyTorch is installed for your Python environment and that your system has enough RAM/storage for model loading.

### Slow CPU execution

This project can run on CPU, but model loading and generation may be slow. A compatible GPU can improve performance.

### Model download issue

Ensure internet access is available during the first model download. Hugging Face models may be cached locally after downloading.

## Reproducibility

Run the notebook from the first cell to the last cell so that embeddings, document chunks, FAISS, the Qwen pipeline, retriever, and RAG chain are initialized in the expected order.
