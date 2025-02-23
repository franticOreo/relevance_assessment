# Insurance Policy RAG System

A Retrieval-Augmented Generation (RAG) system specialized for querying and comparing insurance policy documents, built as part of the Relevance AI take-home challenge. The system implements multiple RAG pipelines with different retrieval and generation strategies for comparative evaluation.

## Challenge Overview
Build and evaluate multiple RAG pipelines for insurance policy document analysis within a 2-hour timeframe:
1. Create retrieval frameworks (hybrid search vs vector search)
2. Create generation frameworks
3. Create agentic frameworks
4. Create evaluation framework

## Implementation Approach

### Document Processing
- Chunking Strategy: Split by title for maintaining context completeness
- Document Sources: NRMA and Allianz insurance policy documents
- Processing Method: UnstructuredLoader with title-based chunking

### Retrieval Methods
- BM25 (Sparse Retrieval)
- Dense Vector Retrieval with OpenAI Embeddings
- Hybrid Retrieval (combining BM25 and Dense)

**Reasoning for retrieval choices:**
- Hybrid approach combines lexical matching (BM25) with semantic understanding (dense vectors)
- BM25 excels at exact term matching crucial for insurance terminology
- Vector search captures semantic relationships and context
- Weighted combination (α=0.5) balances both approaches

### Generation Methods
1. Standard RAG Pipeline
   - Direct question-answering
   - Single-step generation
   - Optimized for precise, factual responses

2. Agentic RAG Pipeline
   - Dynamic approach selection
   - Short-form vs comprehensive analysis responses
   - Built using LangGraph for state management
   - Decision nodes for response strategy selection

**Pipeline Selection Logic:**
- Query complexity analysis
- Length and detail requirements detection
- Domain-specific keyword recognition

## Evaluation Framework

### Test Dataset Generation
1. Ragas-based test generation
2. Knowledge Graph-guided query generation
   - Entity extraction (NER)
   - Key phrase identification
   - Relationship mapping using Jaccard similarity
3. Manual expert-crafted test cases

### Metrics
1. Retrieval Performance:
   - Precision@K
   - Recall@K
   - Mean Reciprocal Rank (MRR)
   - Retrieval Latency

2. Generation Quality:
   - ROUGE-L Score
   - BLEU Score
   - LLM-based Answer Quality Assessment

3. Composite Scoring:
   - Weighted combination of retrieval and generation metrics
   - Normalized scores for fair comparison

## Future Improvements
1. Implement additional embedding models (Cohere, GTE-Small)
2. Add HyDE (Hypothetical Document Embeddings)
3. Expand test dataset with more edge cases


## Getting Started

src/rag/
├── chains/ # Chain implementations
├── embeddings/ # Embedding models
├── evaluation/ # Evaluation framework
├── generation/ # LLM implementations
├── loader/ # Document loading
├── retrieval/ # Retrieval methods
└── main.py # Entry point

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




