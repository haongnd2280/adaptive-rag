import yaml
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

from utils import trace
from tools import retriever
from components.state import State


file_path = "../llm_config.yaml"
with open(file_path, "r") as f:
    llm_config = yaml.safe_load(f)

# LLM
llm = ChatOpenAI(
    model_name=llm_config["filter"]["model"],
    temperature=llm_config["filter"]["temperature"],
)

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


@trace("node")
def filter(state: State) -> State:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {
                "question": question,
                "document": doc.page_content
            }
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(doc)

    return {
        "question": question,
        "documents": filtered_docs
    }


if __name__ == "__main__":
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

    # print(docs)
    print(len(docs))
    print(doc_txt)
