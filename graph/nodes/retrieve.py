from typing import Dict, Any
from graph.state import GraphState
from ingestion import retriever

def retrieve(state:GraphState) -> Dict[str, Any]:
    #This is the retrieve node, a function
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}   #LangGraph merges these outputs into the full state internally.

#In LangGraph, it's common to:
#Use GraphState (a TypedDict) to define the entire state structure.
#Have each node return a partial state update using Dict[str, Any].








