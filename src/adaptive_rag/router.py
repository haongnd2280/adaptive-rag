### Router

from typing import Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # TODO: Add a value indicating the model should answer the question by itself

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


if __name__ == "__main__":
    print(
        question_router.invoke(
            {"question": "Who will the Bears draft first in the NFL draft?"}     # datasource: web_search
        )
    )
    print(
        question_router.invoke(
            {"question": "What are the types of agent memory?"}                  # datasource: vectorstore
        )
    )

    print(
        question_router.invoke(
            {"question": "What is the capital of Vietnam?"}                     # datasource: web_search (not a good choice)
        )
    )
