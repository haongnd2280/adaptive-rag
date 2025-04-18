### Hallucination Grader

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from adaptive_rag.index import retriever
from adaptive_rag.grader import retrieval_grader
from adaptive_rag.generate import rag_chain


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


if __name__ == "__main__":
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    grade = retrieval_grader.invoke({"question": question, "document": doc_txt})

    if grade.binary_score == "yes":
        generation = rag_chain.invoke({"context": docs, "question": question})
        print(generation)
        
        hallu_grade = hallucination_grader.invoke({"documents": docs, "generation": generation})
        print(hallu_grade.binary_score)