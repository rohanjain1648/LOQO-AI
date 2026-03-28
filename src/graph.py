"""
LangGraph assembly — wires all agents together with sequential flows,
parallel steps, conditional edges, and retry routing.

This is the core orchestration file for the entire pipeline.

WORKFLOW:
    START → extract_article → generate_narration
        → [plan_visuals ∥ generate_headlines]  (parallel)
        → review_quality
        → CONDITIONAL: finalize | retry_editor | retry_visual | retry_headline | force_finalize
        → format_output → END
"""

from langgraph.graph import StateGraph, START, END
from src.state import BroadcastState
from src.agents.extraction import extraction_agent
from src.agents.editor import editor_agent
from src.agents.visual import visual_agent
from src.agents.headline import headline_agent
from src.agents.qa import qa_agent
from src.output.formatter import format_final_output


def route_after_qa(state: BroadcastState) -> str:
    """
    Conditional edge router — decides what happens after QA evaluation.
    
    This function is called by LangGraph after the review_quality node.
    It reads the QA results from state and returns a routing key.
    
    Routing logic:
        1. qa_pass=True → "finalize" (skip retries, go to output)
        2. retry_count >= 3 → "force_finalize" (max retries, best effort)
        3. "editor" in targets → "retry_editor" (highest priority)
        4. "visual" in targets → "retry_visual"
        5. "headline" in targets → "retry_headline"
        6. fallback → "finalize"
    
    Single-target priority: editor > visual > headline.
    Rationale: Editor output feeds both Visual and Headline agents,
    so fixing narration first often cascades fixes downstream.
    """
    # Path 1: All criteria pass → finalize immediately
    if state.get("qa_pass", False):
        return "finalize"

    # Path 2: Max retries reached → force-finalize with best effort
    if state.get("retry_count", 0) >= 3:
        return "force_finalize"

    # Path 3-5: Route to highest-priority failing agent
    targets = state.get("qa_failure_targets", [])

    if "editor" in targets:
        return "retry_editor"
    elif "visual" in targets:
        return "retry_visual"
    elif "headline" in targets:
        return "retry_headline"

    # Safety fallback
    return "finalize"


def build_graph():
    """
    Constructs and compiles the full LangGraph workflow.
    
    Returns a compiled LangGraph app ready to be invoked with
    {"article_url": url, "retry_count": 0, "errors": []}.
    """
    workflow = StateGraph(BroadcastState)

    # ══════════════════════════════════════════════
    # Register all nodes
    # ══════════════════════════════════════════════
    workflow.add_node("extract_article", extraction_agent)
    workflow.add_node("generate_narration", editor_agent)
    workflow.add_node("plan_visuals", visual_agent)
    workflow.add_node("generate_headlines", headline_agent)
    workflow.add_node("review_quality", qa_agent)
    workflow.add_node("format_output", format_final_output)

    # ══════════════════════════════════════════════
    # Sequential edges: START → Extraction → Editor
    # ══════════════════════════════════════════════
    workflow.add_edge(START, "extract_article")
    workflow.add_edge("extract_article", "generate_narration")

    # ══════════════════════════════════════════════
    # Parallel fan-out: Editor → [Visual, Headline]
    # Both edges from same source = parallel execution
    # in the same LangGraph superstep
    # ══════════════════════════════════════════════
    workflow.add_edge("generate_narration", "plan_visuals")
    workflow.add_edge("generate_narration", "generate_headlines")

    # ══════════════════════════════════════════════
    # Fan-in: [Visual, Headline] → QA
    # LangGraph waits for BOTH to complete before
    # running review_quality
    # ══════════════════════════════════════════════
    workflow.add_edge("plan_visuals", "review_quality")
    workflow.add_edge("generate_headlines", "review_quality")

    # ══════════════════════════════════════════════
    # Conditional edges: QA → next step
    # 5 possible paths based on evaluation results
    # ══════════════════════════════════════════════
    workflow.add_conditional_edges(
        "review_quality",
        route_after_qa,
        {
            "finalize": "format_output",
            "retry_editor": "generate_narration",
            "retry_visual": "plan_visuals",
            "retry_headline": "generate_headlines",
            "force_finalize": "format_output",
        },
    )

    # ══════════════════════════════════════════════
    # Terminal edge: Output → END
    # ══════════════════════════════════════════════
    workflow.add_edge("format_output", END)

    return workflow.compile()


# ── Compiled app, ready for import ──
app = build_graph()
