import yaml

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

from utils import trace
from components.state import State


file_path = "../llm_config.yaml"
with open(file_path, "r") as f:
    llm_config = yaml.safe_load(f)

# LLM
llm = ChatOpenAI(
    model_name=llm_config["transform"]["model"],
    temperature=llm_config["transform"]["temperature"],
)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

rewriter = re_write_prompt | llm | StrOutputParser()


@trace("node")
def transform(state: State) -> State:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question
    }


if __name__ == "__main__":
    question = "memory agent"
    print(rewriter.invoke({"question": question}))
