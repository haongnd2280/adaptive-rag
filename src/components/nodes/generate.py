import yaml

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

from utils import trace
from components.state import State


file_path = "../llm_config.yaml"
with open(file_path, "r") as f:
    llm_config = yaml.safe_load(f)

# LLM
llm = ChatOpenAI(
    model_name=llm_config["generate"]["model"],
    temperature=llm_config["generate"]["temperature"],
)

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()


@trace("node")
def generate(state: State) -> State:
    """
    Generate answer based on the question and filterd retrieved documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke(
        {
            "context": documents,
            "question": question
        }
    )

    return {
        "question": question,
        "documents": documents,
        "generation": generation
    }
