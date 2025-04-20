from utils import trace
from tools import retriever
from components.state import State


@trace("node")
def retrieve(state: State) -> State:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {
        "question": question,
        "documents": documents
    }
