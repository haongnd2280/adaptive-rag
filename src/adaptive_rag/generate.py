### Generate

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

from adaptive_rag.index import retriever
from adaptive_rag.grader import retrieval_grader


# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    # Run
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    score = retrieval_grader.invoke({"question": question, "document": doc_txt})

    if score.binary_score == "yes":
        generation = rag_chain.invoke({"context": docs, "question": question})
        print(generation)
