from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("===WEB SEARCH===")
    question = state["question"]
    if "documents" in state:
        documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})["results"]
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    if ("documents" in state) and (documents is not None):
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}