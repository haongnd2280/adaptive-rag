from langchain.schema import Document

from utils import trace
from tools import web_search_tool

@trace("node")
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


if __name__ == "__main__":
    # Example usage
    state = {"question": "What is the capital of France?"}
    result = web_search(state)
    print(result)

    