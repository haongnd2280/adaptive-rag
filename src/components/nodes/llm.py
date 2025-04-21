import yaml

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils import trace
from components.state import State

from dotenv import load_dotenv
load_dotenv()


file_path = "../llm_config.yaml"
with open(file_path, "r") as f:
    llm_config = yaml.safe_load(f)

# LLM
llm = ChatOpenAI(
    model_name=llm_config["common"]["model"],
    temperature=llm_config["common"]["temperature"],
)


# Prompt
system = """Given an user's common / general question, use your training knowledge to answer the question."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

common_llm = prompt | llm


@trace("node")
def llm(state: State, llm: ChatOpenAI = common_llm) -> State:
    """LLM node.

    Args:
        state (State): State of the graph.

    Returns:
        State: Updated state.
    """
    # Get the user question from the state
    question = state["question"]

    # Run the LLM with the prompt and get the answer
    generation = llm.invoke({"question": question})

    return {
        "question": question,
        "generation": generation,
    }
