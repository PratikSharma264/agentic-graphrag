from pathlib import Path
import subprocess

from google.adk import Agent

GRAPH_ROOT = Path(__file__).resolve().parent.parent / "graphrag_workspace"

def _run_graphrag(question: str, method: str) -> dict:
    completed = subprocess.run(
        [
            "graphrag",
            "query",
            question,
            "--root",
            str(GRAPH_ROOT),
            "--method",
            method,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if completed.returncode != 0:
        return {
            "status": "error",
            "method": method,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }

    return {
        "status": "success",
        "method": method,
        "report": completed.stdout.strip(),
    }

def search_incident_entity(question: str) -> dict:
    """Search a specific incident, service, team, cause, or relationship.

    Use this for questions centered on a named entity or one incident.
    """
    return _run_graphrag(question, "local")



def analyze_incident_patterns(question: str) -> dict:
    """Analyze trends and themes across the complete incident collection.

    Use this for recurring causes, comparisons, patterns, and whole-corpus questions.
    """
    return _run_graphrag(question, "global")


def search_raw_incident_text(question: str) -> dict:
    """Search source text for exact wording or details missing from graph summaries.

    Use this for quotation requests, direct source evidence, or fallback retrieval.
    """
    return _run_graphrag(question, "basic")


root_agent = Agent(
    name="incident_graph_agent",
    model="gemini-3.5-flash-lite",
    description="Investigates synthetic production incidents with GraphRAG.",
    instruction="""
You are an incident investigation agent. Always use at least one GraphRAG tool
before answering a factual incident question.

Use search_incident_entity for a named incident, service, team, cause, or
relationship. Use analyze_incident_patterns for trends, comparisons, recurring
causes, shared customer effects, monitoring gaps, or questions about the entire
incident collection. Use search_raw_incident_text for quotations, exact source
wording, or direct details that graph summaries do not surface. It may also be
used after another tool returns an error or insufficient evidence.

Base the answer on tool results. Do not invent incident facts. If every selected
tool returns an error, explain the errors instead of answering from memory.
Begin every successful answer with exactly one of these lines, using the method
that supplied the final evidence:
Search method: local
Search method: global
Search method: basic
""",
    tools=[
        search_incident_entity,
        analyze_incident_patterns,
        search_raw_incident_text,
    ],
)