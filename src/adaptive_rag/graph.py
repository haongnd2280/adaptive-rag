from langgraph.graph import END, StateGraph, START

from components.state import State
from components.nodes import (
    web_search,
    retrieve,
    filter,
    generate,
    transform,
)
from components.edges import (
    router,
    decide,
    grade,
)


workflow = StateGraph(State)

# Define nodes
workflow.add_node("web_search", web_search)       # web search
workflow.add_node("retrieve", retrieve)           # retrieve
workflow.add_node("filter", filter)               # filter documents
workflow.add_node("generate", generate)           # generatae
workflow.add_node("transform", transform)         # transform_query

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
workflow.add_edge("retrieve", "filter")
workflow.add_conditional_edges(
    "filter",
    decide,
    {
        "transform": "transform",
        "generate": "generate",
    },
)
workflow.add_edge("transform", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade,
    {
        "hallucination": "generate",
        "ok": END,
        "not ok": "transform",
    },
)

# Compile
app = workflow.compile()
