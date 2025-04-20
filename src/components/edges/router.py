from typing import Literal

import yaml
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

from utils import trace
from components.state import State


file_path = "../llm_config.yaml"
with open(file_path, "r") as f:
    llm_config = yaml.safe_load(f)

llm = ChatOpenAI(
    model_name=llm_config["router"]["name"],
    temperature=llm_config["router"]["temperature"],
)

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # TODO: Add a value indicating the model should answer the question by itself

    data_src: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Create runnable
question_router = route_prompt | structured_llm_router


@trace("edge")
def router(state: State, verbose: bool = True) -> Literal["web_search", "vectorstore"]:
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.data_src == "web_search":
        src = "web_search"
    elif source.data_src == "vectorstore":
        src = "vectorstore"

    if verbose:
        print(f"Routing question '{question}' to {src}")

    return src


if __name__ == "__main__":
    print(
        question_router.invoke(
            {"question": "Who will the Bears draft first in the NFL draft?"}     # datasource: web_search
        )
    )
    print(
        question_router.invoke(
            {"question": "What are the types of agent memory?"}                  # datasource: vectorstore
        )
    )

    print(
        question_router.invoke(
            {"question": "What is the capital of Vietnam?"}                     # datasource: web_search (not a good choice)
        )
    )
