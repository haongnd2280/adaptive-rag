from typing import Literal

from utils import trace
from components.state import State


@trace("edge")
def decide(state: State, verbose: bool = True) -> Literal["generate", "transform"]:
    """
    Determines whether to generate an answer, or re-generate a question
    based on the number of relevant documents.

    - If no documents are relevant, we will transform the question.
    - If there are relevant documents, we will generate an answer.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        decision = "transform"
    else:
        decision = "generate"

    if verbose:
        if decision == "transform":
            print("All documents are not relevant to the question --> transform query...")
        elif decision == "generate":
            print("Decision: Generate answer...")

    return decision
