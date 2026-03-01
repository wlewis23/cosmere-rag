#scripts/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 
import gradio as gr
from src.rag.chain import ask



def chat(message, history):
    answer, docs = ask(message)
    sources = sorted(set(doc.metadata.get("source", "Unknown") for doc in docs))
    #sources_md = "\n\n**Sources:** " + " · ".join(f"`{s}`" for s in sources)

    #attempt at source hyperlink
    sources_md = "\n\n**Sources:** " + " · ".join(f"[{s}](https://coppermind.net/wiki/{s.replace(' ', '_')})" for s in sources)
    return answer + sources_md

gr.ChatInterface(
    chat,
    title="Cosmere RAG",
    description="Ask anything about Brandon Sanderson's Cosmere Universe.",
    #Set share=True to create share link. Probably will require firewall changes. 
    ).launch(share=False)