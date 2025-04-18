### Answer Grader

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from adaptive_rag.index import retriever
from adaptive_rag.grader import retrieval_grader
from adaptive_rag.generate import rag_chain
from adaptive_rag.hallu_grader import hallucination_grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


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

        if hallu_grade.binary_score == "no":
            answer_grade = answer_grader.invoke({"question": question, "generation": generation})
            print(answer_grade.binary_score)
        else:
            print("Answer is not grounded in the facts.")