"""
demo.py — Non-interactive demo of the Employee Q&A Agent
Run: python demo.py
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not set in .env"); sys.exit(1)

from rag.document_loader import load_sample_documents
from rag.vectorstore import initialize_vectorstore
from agent.graph import build_graph
from main import run_question

DEMO_QUESTIONS = [
    "How do I set up VPN access?",
    "What is the meal expense limit when travelling?",
    "How many vacation days do I get after 4 years of service?",
]

print("\nLoading knowledge base …")
docs = load_sample_documents()
vs   = initialize_vectorstore(docs)
graph = build_graph(vs)
print("Agent ready.\n")

for i, q in enumerate(DEMO_QUESTIONS, 1):
    print(f"\n{'#'*70}")
    print(f"  DEMO QUESTION {i}/{len(DEMO_QUESTIONS)}")
    print(f"{'#'*70}")
    run_question(graph, q, thread_id=f"demo-{i}")
    input("\n  Press Enter for next question …\n") if i < len(DEMO_QUESTIONS) else None

print("\nDemo complete!")
