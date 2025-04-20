from langgraph.graph import END, StateGraph, START

from components.state import State
from components.nodes import (
    web_search,
    retrieve,
    grade_documents,
    generate,
    transform,
)
from components.edges import (
    router,
    decide_generation,
    grade_generation,
)


workflow = StateGraph(State)

# Define nodes
workflow.add_node("web_search", web_search)            # web search
workflow.add_node("retrieve", retrieve)                # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)                # generatae
workflow.add_node("transform", transform)              # transform_query

# Define edges
# First edge is an router
workflow.add_conditional_edges(
    START,
    router,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
# web_search based generation will not be checked
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_generation,
    {
        "transform": "transform",
        "generate": "generate",
    },
)
workflow.add_edge("transform", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "hallucination": "generate",
        "ok": END,
        "not ok": "transform",
    },
)

# Compile
app = workflow.compile()
