"""
Agent 4: QA / Evaluation Agent

Scores the assembled broadcast screenplay against 5 criteria (1-5 each)
and 6 boolean checks. Determines pass/fail and routes targeted retries
to specific failing agents.

This is the decision-making hub that drives conditional edges in LangGraph.
"""

import json
from src.state import BroadcastState
from src.config import get_llm
from src.models.schemas import QAResult
from src.prompts.qa_prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE


def qa_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Evaluates the screenplay and decides next route.

    Reads: source_title, article_text, segments, visual_plan, headline_plan,
           total_duration_sec, retry_count
    Writes: qa_scores, qa_checks, qa_pass, qa_failure_targets, qa_feedback,
            editor_retry_feedback, visual_retry_feedback, headline_retry_feedback,
            retry_count, current_route
    """
    errors = []

    segments = state.get("segments", [])
    visual_plan = state.get("visual_plan", [])
    headline_plan = state.get("headline_plan", [])
    retry_count = state.get("retry_count", 0)

    # ── Handle empty state gracefully ──
    if not segments:
        return {
            "qa_scores": _default_scores(),
            "qa_checks": _default_checks(),
            "qa_pass": False,
            "qa_failure_targets": ["editor"],
            "qa_feedback": "No narration segments were generated. Editor must retry.",
            "editor_retry_feedback": "No segments were generated. Please generate 4-6 narration segments.",
            "visual_retry_feedback": "",
            "headline_retry_feedback": "",
            "retry_count": retry_count + 1,
            "current_route": "retry_editor",
            "errors": ["QA: No segments to evaluate."],
        }

    # ── Build user prompt with all data ──
    user_prompt = QA_USER_TEMPLATE.format(
        source_title=state.get("source_title", "Unknown"),
        article_text=state.get("article_text", "")[:3000],  # Truncate to save tokens
        segments_json=json.dumps(segments, indent=2),
        visual_plan_json=json.dumps(visual_plan, indent=2),
        headline_plan_json=json.dumps(headline_plan, indent=2),
        total_duration_sec=state.get("total_duration_sec", 0),
        retry_count=retry_count,
    )

    # ── Call Gemini for evaluation ──
    try:
        llm = get_llm(temperature=0.2)  # Low temp for analytical scoring
        structured_llm = llm.with_structured_output(QAResult)

        result = structured_llm.invoke(
            [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        # ── Extract scores and checks ──
        scores = result.scores.model_dump()
        checks = result.checks.model_dump()

        # ── Recompute pass/fail using our own logic (don't trust LLM) ──
        qa_pass = _evaluate_pass(scores, checks)
        failure_targets = _determine_failure_targets(scores, checks)

        # ── If LLM says pass but our logic disagrees, use our logic ──
        if result.overall_pass and not qa_pass:
            errors.append(
                "QA: LLM declared pass but programmatic check failed. Using stricter check."
            )

        # ── Build per-agent retry feedback ──
        editor_feedback = result.editor_feedback or ""
        visual_feedback = result.visual_feedback or ""
        headline_feedback = result.headline_feedback or ""

        return {
            "qa_scores": scores,
            "qa_checks": checks,
            "qa_pass": qa_pass,
            "qa_failure_targets": failure_targets,
            "qa_feedback": result.feedback,
            "editor_retry_feedback": editor_feedback if "editor" in failure_targets else "",
            "visual_retry_feedback": visual_feedback if "visual" in failure_targets else "",
            "headline_retry_feedback": headline_feedback if "headline" in failure_targets else "",
            "retry_count": retry_count + 1,
            "current_route": "finalize" if qa_pass else _get_route(failure_targets),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"QA agent error: {str(e)}")

        # On error, pass through to avoid infinite loops
        return {
            "qa_scores": _default_scores(),
            "qa_checks": _default_checks(),
            "qa_pass": True,  # Force pass on QA error to prevent loops
            "qa_failure_targets": [],
            "qa_feedback": f"QA evaluation failed with error: {str(e)}. Passing by default.",
            "editor_retry_feedback": "",
            "visual_retry_feedback": "",
            "headline_retry_feedback": "",
            "retry_count": retry_count + 1,
            "current_route": "finalize",
            "errors": errors,
        }


def _evaluate_pass(scores: dict, checks: dict) -> bool:
    """
    Programmatic pass/fail evaluation.
    
    Rules:
    1. No criterion score below 3
    2. Overall average >= 4.0
    3. All boolean checks must be True
    """
    # Rule 1: No score below 3
    for criterion, score in scores.items():
        if score < 3:
            return False

    # Rule 2: Average >= 4.0
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    if avg_score < 4.0:
        return False

    # Rule 3: All boolean checks pass
    for check_name, passed in checks.items():
        if not passed:
            return False

    return True


def _determine_failure_targets(scores: dict, checks: dict) -> list:
    """
    Determine which agents are responsible for failures.
    
    Responsibility mapping:
    - editor: story_structure, hook_engagement, narration_quality, 
              duration_fit, coverage, factual_grounding
    - visual: visual_planning, timeline_coherence
    - headline: headline_quality, text_fit, redundancy_free
    """
    targets = []

    # Check Editor responsibility
    editor_fails = (
        scores.get("story_structure", 5) < 3 or
        scores.get("hook_engagement", 5) < 3 or
        scores.get("narration_quality", 5) < 3 or
        not checks.get("duration_fit", True) or
        not checks.get("coverage", True) or
        not checks.get("factual_grounding", True)
    )
    if editor_fails:
        targets.append("editor")

    # Check Visual responsibility
    visual_fails = (
        scores.get("visual_planning", 5) < 3 or
        not checks.get("timeline_coherence", True)
    )
    if visual_fails:
        targets.append("visual")

    # Check Headline responsibility
    headline_fails = (
        scores.get("headline_quality", 5) < 3 or
        not checks.get("text_fit", True) or
        not checks.get("redundancy_free", True)
    )
    if headline_fails:
        targets.append("headline")

    # If average is low but no specific criterion is below 3,
    # retry the editor (since narration improvement cascades)
    if not targets:
        avg = sum(scores.values()) / len(scores) if scores else 0
        if avg < 4.0:
            targets.append("editor")

    return targets


def _get_route(failure_targets: list) -> str:
    """Get routing decision based on failure targets (priority order)."""
    if "editor" in failure_targets:
        return "retry_editor"
    elif "visual" in failure_targets:
        return "retry_visual"
    elif "headline" in failure_targets:
        return "retry_headline"
    return "finalize"


def _default_scores() -> dict:
    """Default scores when QA can't evaluate."""
    return {
        "story_structure": 1,
        "hook_engagement": 1,
        "narration_quality": 1,
        "visual_planning": 1,
        "headline_quality": 1,
    }


def _default_checks() -> dict:
    """Default checks when QA can't evaluate."""
    return {
        "factual_grounding": False,
        "coverage": False,
        "duration_fit": False,
        "text_fit": False,
        "redundancy_free": False,
        "timeline_coherence": False,
    }
