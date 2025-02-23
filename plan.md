Plan of delivery:
* let me know if this is bad idea, I was thinking if I could just run this entire project as a docker container (not sure where it would be running), so the assesor can simply plug into my project and query it.
* I was thinking the docker container could run the assessor could intferface via cli.
-- e.g choose a rag method: vanilla rag, agentic rag
-- query documents with a question.


Plan for development:
1. Evaluate the Dense Retriever in Isolation
Goal: Compare its standalone performance against BM25 and the Hybrid version.
Action Items:
Create a small script (or function) to measure the retrieval accuracy of the dense retriever alone on your test set.
Use part of your existing judge logic or a simpler metric (like recall@k) before diving into the Hybrid comparison.
2. Extend Hugging Face LLM in huggingface_llm.py
Goal: Implement a second pipeline (e.g., “LLM #2”) parallel to the current OpenAI-based approach.
Options:
LLaMA-based model: For instance, if there’s a “Llama3” or a similarly open and free to serve model.
Alternatively, use a lighter tech QA–focused model that can handle up to ~100 pages of PDF efficiently.
Action Items:
Instantiate the Hugging Face pipeline using the relevant model.
If you only need short answers, you might configure the generation parameters (like max tokens, temperature) specifically for summarization or QA.
3. Implement and Use metric_calculations.py
Goal: Quantify performance for different pipeline permutations.
Action Items:
Generate a test set (possibly with your existing code that uses ragas.testset or your own queries).
Save these queries to a small local file or a DataFrame for reproducibility.
Using judge_llm.py or your own numeric metrics, gather scores for each pipeline permutation:
BM25 + GPT
Dense + GPT
Hybrid + GPT
Hybrid + Your new Hugging Face model
(Optionally) Dense + Your new Hugging Face model
Plot the results (matplotlib) and export as PNG for your README.
4. Add More Embeddings (e.g., Cohere or GTE-Small)
Goal: Expand your retrieval toolkit.
References:
Cohere Embeddings
GTE-Small – a candidate for compact yet effective embeddings.
Action Items:
Implement a helper (similar to get_cohere_embeddings) for GTE-Small.
Test out performance differences (BM25 vs. GTE embeddings vs. OpenAI embeddings).
5. Experiment with HyDE
Goal: Incorporate Hypothetical Document Embeddings (HyDE) to see if it boosts zero-shot retrieval performance.
Reference: HyDE Paper
Action Items:
Integrate a step that generates a “hypothetical” doc for each query using an instruction-following model (like GPT or another LLM).
Encode that hypothetical doc with your chosen encoder (e.g., GTE-Small, Cohere, etc.).
Retrieve actual docs from your corpus based on vector similarity.
Compare results to your baseline retrieval pipelines.

left over:
1. check eval metrics works.
2. how to handle env vars.