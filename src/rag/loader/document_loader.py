from langchain_unstructured import UnstructuredLoader
from typing import List
from langchain.schema import Document

def load_insurance_docs(file_paths: List[str]) -> List[Document]:
    """
    Load multiple insurance documents using UnstructuredLoader.
    
    Args:
        file_paths: List of paths to PDF documents
        
    Returns:
        List of loaded Document objects
    """
    documents = []
    for path in file_paths:
        loader = UnstructuredLoader(
            path,
            chunking_strategy="by_title"
        )
        documents.extend(loader.load())
    return documents

# Default insurance document paths
DEFAULT_DOCS = [
    "rag/docs/nrma.pdf",
    "rag/docs/allianz.pdf"
]

def get_default_documents() -> List[Document]:
    """Helper function to load the default insurance documents."""
    return load_insurance_docs(DEFAULT_DOCS)