from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state):
    print("=====ASSESS GRADED DOCUMENTS=====")
    if state["web_search"]:
        print("===Decision: Not all tdocuments are relevant to the question, including web search...")
        return WEBSEARCH
    else:
        print("===Decision: All documents are relevant, generating answer...")
        return GENERATE
    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, path_map={WEBSEARCH: WEBSEARCH, GENERATE:GENERATE})  #path_map is optional. Maps the output of the conditinal function with the names of next nodes

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
