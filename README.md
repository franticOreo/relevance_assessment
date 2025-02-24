# Insurance Policy RAG System

A Retrieval-Augmented Generation (RAG) system specialized for querying and comparing insurance policy documents, built as part of the Relevance AI take-home challenge. The system implements multiple RAG pipelines with different retrieval and generation strategies for comparative evaluation.

This repository provides a fully functional RAG setup, referencing internal modules for document loading, retrieval, generation, and evaluation.

---

## Challenge Overview
Build and evaluate multiple RAG pipelines for insurance policy document analysis within a 2-hour timeframe:
1. Create retrieval frameworks (hybrid search vs vector search)
2. Create generation frameworks
3. Create agentic frameworks
4. Create evaluation framework

---

## Implementation Approach

### Document Processing
- Chunking Strategy: Split by title for maintaining context completeness. This chunking logic is demonstrated in the document loading utility (see "src/rag/loader/document_loader.py").
- Document Sources: NRMA and Allianz insurance policy documents
- Processing Method: UnstructuredLoader with title-based chunking

### Retrieval Methods
- BM25 (Sparse Retrieval) – Managed by the BM25Retriever class (see "src/rag/retrieval/bm25_retriever.py")
- Dense Vector Retrieval with OpenAI Embeddings – Handled by the VectorRetriever component (see "src/rag/retrieval/vector_retriever.py")
- Hybrid Retrieval (combining BM25 and Dense) – Implemented via the HybridRetriever (see "src/rag/retrieval/hybrid_retriever.py")

**Reasoning for retrieval choices:**
- Hybrid approach combines lexical matching (BM25) with semantic understanding (dense vectors).
- BM25 excels at exact term matching crucial for insurance terminology.
- Vector search captures semantic relationships and context.
- Weighted combination (α=0.5) balances both approaches.

### Generation Methods
1. Standard RAG Pipeline
   - Uses the StandardRAG implementation (see "src/rag/generation/standard_rag.py") for direct question-answering, focusing on a single-step generation flow.

2. Agentic RAG Pipeline
   - Defined in the AgenticRAG class (see "src/rag/generation/agentic_rag.py").
   - Dynamically selects the approach (short-form vs comprehensive) based on the query's complexity.
   - Built using a simple graph-based workflow for decision-making.

**Pipeline Selection Logic:**
- Query complexity analysis.
- Length and detail requirements detection.
- Domain-specific keyword recognition.

---

## Evaluation Framework

### Test Dataset Generation
1. Ragas-based test generation.  
2. Knowledge Graph-guided query generation (entity extraction, key phrase identification, relationship mapping).  
3. Manual expert-crafted test cases.

### Metrics
1. Retrieval Performance: Precision@K, Recall@K, MRR, Retrieval Latency.
2. Generation Quality: ROUGE-L Score, BLEU Score, LLM-based Answer Quality Assessment.  
3. Composite Scoring: Weighted combination of retrieval and generation metrics, normalized for fair comparison.

This evaluation logic is implemented in "src/rag/evaluation/metric_calculations.py" and orchestrated by the PipelineEvaluator in "src/rag/evaluation/pipeline_evaluator.py".

---

## Future Improvements
1. Implement additional embedding models (Cohere, GTE-Small) (see "src/rag/embeddings/cohere_embeddings.py" for a partial reference).  
2. Add HyDE (Hypothetical Document Embeddings).  
3. Expand test dataset with more edge cases.

---

## Getting Started
# Quick Overview of File Structure

Below is a concise look at the main directories under `src/rag/`:

- **chains/**  
  - Contains specialized *chain* classes (e.g., `LongQAChain` for handling multi-document QA logic).

- **embeddings/**  
  - Stores various embedding providers (e.g., `openai_embeddings.py`, `cohere_embeddings.py`) responsible for generating vector representations of documents.

- **evaluation/**  
  - Houses the evaluation framework (e.g., `pipeline_evaluator.py`, `metric_calculations.py`) that calculates retrieval and generation metrics, orchestrates test datasets, and provides scoring logic.

- **generation/**  
  - Implements different approaches for RAG-based generation (e.g., `standard_rag.py` for a straightforward pipeline, `agentic_rag.py` for more dynamic or “agentic” reasoning flows).

- **loader/**  
  - Manages document loading utilities (e.g., `document_loader.py` for reading raw PDFs or text and splitting them into workable chunks).

- **retrieval/**  
  - Implements retrieval mechanisms (e.g., `bm25_retriever.py`, `vector_retriever.py`, `hybrid_retriever.py`) that fetch relevant documents from the knowledge corpus.

- **main.py**  
  - The entry point that sets up the overall Retrieval-Augmented Generation pipeline, including environment variable loading, retriever creation, and LLM model configuration.

Project file structure:
```
├── rag
│   ├── chains
│   │   └── long_qa_chain.py
│   ├── data
│   │   └── ragas_testset.csv
│   ├── docs
│   │   ├── allianz.pdf
│   │   └── nrma.pdf
│   ├── embeddings
│   │   ├── cohere_embeddings.py
│   │   └── openai_embeddings.py
│   ├── evaluation
│   │   ├── evaluation.ipynb
│   │   ├── evaluation.py
│   │   ├── judge_llm.py
│   │   ├── metric_calculations.py
│   │   ├── pipeline_evaluator.py
│   │   └── testset_generation.py
│   ├── generation
│   │   ├── agentic_rag.py
│   │   ├── base_llm.py
│   │   ├── huggingface_llm.py
│   │   ├── openai_llm.py
│   │   └── standard_rag.py
│   ├── loader
│   │   └── document_loader.py
│   ├── main.py
│   ├── requirements.txt
│   └── retrieval
│       ├── bm25_retriever.py
│       ├── hybrid_retriever.py
│       └── vector_retriever.py
```

### Prerequisites
- Python 3.11+
- API keys for:
  - OpenAI
  - Hugging Face (for alternative models)
  - Cohere (for embeddings)

### Quick Start
```bash
git clone <repository-url>
cd insurance-rag
cp src/rag/.env.example src/rag/.env
```

Edit the `.env` file with your API keys:
```bash
nano src/rag/.env
# Add your keys:
# OPENAI_API_KEY=your_key_here
# HUGGINGFACE_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here
```

Run the script: 
```bash
python src/rag/main.py
```




