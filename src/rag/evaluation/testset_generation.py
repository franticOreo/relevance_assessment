from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.transforms.extractors import NERExtractor, KeyphrasesExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.synthesizers.base import BaseSynthesizer
from ragas.dataset_schema import SingleTurnSample
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from time import sleep
from langchain_unstructured import UnstructuredLoader
from typing import List
from langchain.schema import Document

load_dotenv()

# documents = get_default_documents()

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
    "../docs/nrma.pdf",
    "../docs/allianz.pdf"
]

def get_default_documents() -> List[Document]:
    """Helper function to load the default insurance documents."""
    return load_insurance_docs(DEFAULT_DOCS)


documents = get_default_documents()

# Step 1: Convert your pre-chunked documents into nodes.
# Assume each element in "elements" has a "text" attribute.
nodes = [Node(properties={"page_content": element.page_content}) for element in documents]

# Step 2: Build the Knowledge Graph from nodes.
kg = KnowledgeGraph(nodes=nodes)

# Step 3: Define and apply the transformation pipeline.
# You can run extractors in parallel and then a relationship builder.
ner_extractor = NERExtractor()
key_extractor = KeyphrasesExtractor()
rel_builder = JaccardSimilarityBuilder(property_name="entities", key_name="PER", new_property_name="entity_jaccard_similarity")

transforms = [
    Parallel(ner_extractor, key_extractor),
    rel_builder
]
# Apply the transforms asynchronously to enrich the knowledge graph with rate limiting
for transform in transforms:
    apply_transforms(kg, [transform])
    sleep(2)  # Add delay between transform applications

generator_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", api_key=os.getenv("OPENAI_API_KEY")))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings,  # Fixed: Using the correct embedding model
    knowledge_graph=kg
)

# Filter documents by content length
filtered_docs = [doc for doc in documents if len(doc.page_content) > 200]

# Generate the test dataset with delay
dataset = generator.generate_with_langchain_docs(filtered_docs, testset_size=10)
sleep(2)  # Add delay after generation

# Convert to pandas and save to CSV
df = dataset.to_pandas()
os.makedirs("../data", exist_ok=True)
df.to_csv("rag/data/ragas_testset.csv", index=False)