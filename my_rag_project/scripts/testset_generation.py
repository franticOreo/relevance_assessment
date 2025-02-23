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

# Step 1: Convert your pre-chunked documents into nodes.
# Assume each element in "elements" has a "text" attribute.
nodes = [Node(properties={"page_content": element.page_content}) for element in elements]

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
# Apply the transforms asynchronously to enrich the knowledge graph.
apply_transforms(kg, transforms)



generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

generator = TestsetGenerator(llm=generator_llm, 
                             embedding_model=embedding_model, 
                             knowledge_graph=kg)

elementsf = [el for el in elements if len(el.page_content) > 200]

dataset = generator.generate_with_langchain_docs(elementsf, testset_size=10)