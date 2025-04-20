from typing import TypedDict


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM generation
        documents: list of retrieved documents
    """

    question: str
    generation: str
    documents: list[str]
