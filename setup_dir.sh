#!/usr/bin/env bash

# Create the base project folder
mkdir -p my_rag_project

# Create subdirectories for data
mkdir -p my_rag_project/data/docs
mkdir -p my_rag_project/data/processed

# Create subdirectories for embeddings
mkdir -p my_rag_project/embeddings
touch my_rag_project/embeddings/cohere_embeddings.py
touch my_rag_project/embeddings/openai_embeddings.py

# Create subdirectories for generation
mkdir -p my_rag_project/generation
touch my_rag_project/generation/openai_llm.py
touch my_rag_project/generation/huggingface_llm.py

# Create subdirectories for retrieval
mkdir -p my_rag_project/retrieval
touch my_rag_project/retrieval/bm25_retriever.py
touch my_rag_project/retrieval/hybrid_retriever.py
touch my_rag_project/retrieval/vector_retriever.py

# Create subdirectories for evaluation
mkdir -p my_rag_project/evaluation
touch my_rag_project/evaluation/judge_llm.py
touch my_rag_project/evaluation/metric_calculations.py

# Create main.py and requirements.txt
touch my_rag_project/main.py
touch my_rag_project/requirements.txt

echo "Project structure created successfully."