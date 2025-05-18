from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

app.get_graph().draw_mermaid_png(output_file_path="CRAG_graph.png")

if __name__ == "__main__":
    print("Holas!")
    response = app.invoke(input={"question":"Explain agent memory and provide an evolution timeline"})
    print(response)


