#pytest . -s -v   #-v: verbose

from dotenv import load_dotenv
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever

from pprint import pprint
from graph.chains.generation import generation_chain

load_dotenv()

def test_retrival_grader_answer_yes() -> None:
    #
    question = "what is an agent memory?"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content #First documents with the highest similarity search

    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})

    assert res.binary_score == "yes"

def test_retrival_grader_answer_no() -> None:
    #
    question = "what is an agent memory?"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content #First documents with the highest similarity search

    res: GradeDocuments = retrieval_grader.invoke({"question": "How to make pizza", "document": doc_txt})

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    #Testing the generatio chain only (Without grading docs or web_search)
    question = "what is an agent memory?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs, "question": question})
    pprint(generation)


if __name__ == "__main__":
    """
    question = "what is an agent memory?"
    docs = retriever.invoke(question)
    doc_0 = docs[0].page_content #First documents with the highest similarity search
    doc_1 = docs[1].page_content
    print("\n\n doc_0 \n\n")
    print(doc_0)
    print("\n\n")
    print(docs[0].metadata)
    print("\n\n doc_1 \n\n")
    print(doc_1)
    print("\n\n")
    print(docs[1].metadata)
    """
    test_generation_chain()