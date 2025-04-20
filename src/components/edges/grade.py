from typing import Literal

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

from utils import trace
from components.state import State


file_path = "../llm_config.yaml"
with open(file_path, "r") as f:
    llm_config = yaml.safe_load(f)

# LLMs
grader_llm = ChatOpenAI(
    model_name=llm_config["answer"]["model"],
    temperature=llm_config["answer"]["temperature"],
)
hallu_llm = ChatOpenAI(
    model_name=llm_config["hallu"]["model"],
    temperature=llm_config["hallu"]["temperature"],
)

def create_answer_grader(llm: ChatOpenAI = grader_llm) -> Runnable:
    # Data model
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

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
    return answer_grader


def create_hallu_grader(llm: ChatOpenAI = hallu_llm) -> Runnable:
    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallu_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallu_grader = hallu_prompt | structured_llm_grader
    return hallu_grader


answer_grader = create_answer_grader()
hallu_grader = create_hallu_grader()


@trace("edge")
def grade(
    state: State,
    verbose: bool = True
) -> Literal["hallucination", "ok", "not ok"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallu_grader.invoke(
        {
            "documents": documents,
            "generation": generation
        }
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        if verbose:
            print("Decision: Generation is grounded in documents.")
            print("Check if generation answers question...")

        score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation
            }
        )
        grade = score.binary_score
        if grade == "yes":
            if verbose:
                print("Decision: Generation addresses the question.")
            return "ok"
        else:
            if verbose:
                print("Decision: Generation does not address the question.")
            return "not ok"
    else:
        if verbose:
            print("Decision: Generation is not grounded in documents. Try to regenerate...")

        return "hallucination"
