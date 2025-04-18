from langgraph.graph import END, StateGraph, START

from adaptive_rag.nodes_edges import (
    State,
    web_search,
    retrieve,
    generate,
    grade_documents,
    transform_query,
    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question,

)

workflow = StateGraph(State)

# Define the nodes
workflow.add_node("web_search", web_search)            # web search
workflow.add_node("retrieve", retrieve)                # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)                # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
# First edge is an router
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",    # if hallucination -> generate again
        "useful": END,
        "not useful": "transform_query",  # answer does not address the question -> rewrite the question
    },
)

# Compile
app = workflow.compile()
