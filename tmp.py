from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)

dotenv.load_dotenv()

# Define your prompt template
template = "Question: {question}\nAnswer: Let's think step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize the Hugging Face model
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Create a chain using the newer pipe syntax
chain = prompt | llm | StrOutputParser()

# Example question
question = "Which planet has the largest moon in the solar system?"
answer = chain.invoke({"question": question})
print(answer)