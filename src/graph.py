"""
LangGraph assembly — wires all agents together with sequential flows,
parallel steps, conditional edges, retry routing, validation gates,
and best-of-N selection.

This is the core orchestration file for the entire pipeline.

WORKFLOW (v2 — Production-Ready Retry System):
    START → extract_article → generate_narration → pre_validate
        → CONDITIONAL: fan_out_parallel | retry_editor
    fan_out_parallel → [plan_visuals ∥ generate_headlines]  (parallel)
        → post_validate
        → CONDITIONAL: review_quality | retry_visual | retry_headline | retry_parallel_gate
    review_quality (LLM QA scoring)
        → CONDITIONAL: select_best | retry_editor | retry_visual | 
                       retry_headline | retry_parallel_gate
    select_best (best-of-N) → format_output → END

RETRY BUDGET: 5 retries per agent (editor, visual, headline independently)
GLOBAL CAP: 8 total QA cycles as safety net
"""

from langgraph.graph import StateGraph, START, END
from src.state import BroadcastState
from src.agents.extraction import extraction_agent
from src.agents.editor import editor_agent
from src.agents.visual import visual_agent
from src.agents.headline import headline_agent
from src.agents.qa import qa_agent
from src.agents.validators import (
    pre_validate_editor,
    post_validate_parallel,
    route_after_pre_validation,
    route_after_post_validation,
)
from src.agents.selector import best_of_n_selector
from src.output.formatter import format_final_output


def route_after_qa(state: BroadcastState) -> str:
    """
    Conditional edge router — reads the route decision from QA agent.
    
    The QA agent handles all budget-aware routing logic internally
    via _smart_route(), storing the result in current_route.
    """
    return state.get("current_route", "select_best")


def fan_out_passthrough(state: BroadcastState) -> dict:
    """Pass-through node that enables parallel fan-out to Visual + Headline."""
    return {}


def retry_parallel_passthrough(state: BroadcastState) -> dict:
    """Pass-through node that enables parallel retry of Visual + Headline."""
    return {}


def build_graph():
    """
    Constructs and compiles the full LangGraph workflow (v2).
    
    Returns a compiled LangGraph app ready to be invoked with:
    {
        "article_url": url,
        "retry_count": 0,
        "errors": [],
        "editor_retry_count": 0,
        "visual_retry_count": 0,
        "headline_retry_count": 0,
        "retry_history": [],
        "retry_decisions": [],
    }
    """
    workflow = StateGraph(BroadcastState)

    # ══════════════════════════════════════════════
    # Register all nodes
    # ══════════════════════════════════════════════
    workflow.add_node("extract_article", extraction_agent)
    workflow.add_node("generate_narration", editor_agent)
    workflow.add_node("pre_validate", pre_validate_editor)
    workflow.add_node("fan_out_parallel", fan_out_passthrough)
    workflow.add_node("plan_visuals", visual_agent)
    workflow.add_node("generate_headlines", headline_agent)
    workflow.add_node("post_validate", post_validate_parallel)
    workflow.add_node("review_quality", qa_agent)
    workflow.add_node("retry_parallel_gate", retry_parallel_passthrough)
    workflow.add_node("select_best", best_of_n_selector)
    workflow.add_node("format_output", format_final_output)

    # ══════════════════════════════════════════════
    # Sequential: START → Extraction → Editor → Pre-Validate
    # ══════════════════════════════════════════════
    workflow.add_edge(START, "extract_article")
    workflow.add_edge("extract_article", "generate_narration")
    workflow.add_edge("generate_narration", "pre_validate")

    # ══════════════════════════════════════════════
    # Pre-validation gate
    # Catches deterministic editor failures before
    # wasting Visual + Headline LLM calls
    # ══════════════════════════════════════════════
    workflow.add_conditional_edges(
        "pre_validate",
        route_after_pre_validation,
        {
            "continue": "fan_out_parallel",
            "retry_editor": "generate_narration",
        },
    )

    # ══════════════════════════════════════════════
    # Parallel fan-out: fan_out_parallel → [Visual, Headline]
    # Both edges from same source = parallel execution
    # in the same LangGraph superstep
    # ══════════════════════════════════════════════
    workflow.add_edge("fan_out_parallel", "plan_visuals")
    workflow.add_edge("fan_out_parallel", "generate_headlines")

    # ══════════════════════════════════════════════
    # Fan-in: [Visual, Headline] → Post-Validate
    # LangGraph waits for BOTH to complete before
    # running post_validate
    # ══════════════════════════════════════════════
    workflow.add_edge("plan_visuals", "post_validate")
    workflow.add_edge("generate_headlines", "post_validate")

    # ══════════════════════════════════════════════
    # Post-validation gate
    # Auto-fixes minor issues, routes major ones
    # back to specific agents
    # ══════════════════════════════════════════════
    workflow.add_conditional_edges(
        "post_validate",
        route_after_post_validation,
        {
            "qa": "review_quality",
            "retry_visual": "plan_visuals",
            "retry_headline": "generate_headlines",
            "retry_parallel": "retry_parallel_gate",
        },
    )

    # ══════════════════════════════════════════════
    # Parallel retry fan-out
    # retry_parallel_gate → [Visual, Headline] simultaneously
    # ══════════════════════════════════════════════
    workflow.add_edge("retry_parallel_gate", "plan_visuals")
    workflow.add_edge("retry_parallel_gate", "generate_headlines")

    # ══════════════════════════════════════════════
    # QA evaluation → conditional routing
    # Supports: select_best, retry_editor, retry_visual,
    #           retry_headline, retry_parallel_gate
    # ══════════════════════════════════════════════
    workflow.add_conditional_edges(
        "review_quality",
        route_after_qa,
        {
            "select_best": "select_best",
            "retry_editor": "generate_narration",
            "retry_visual": "plan_visuals",
            "retry_headline": "generate_headlines",
            "retry_parallel": "retry_parallel_gate",
        },
    )

    # ══════════════════════════════════════════════
    # Best-of-N select → Format output → END
    # ══════════════════════════════════════════════
    workflow.add_edge("select_best", "format_output")
    workflow.add_edge("format_output", END)

    return workflow.compile()


# ── Compiled app, ready for import ──
app = build_graph()
