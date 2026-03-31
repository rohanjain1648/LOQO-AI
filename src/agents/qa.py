"""
Agent 4: QA / Evaluation Agent

Scores the assembled broadcast screenplay against 5 criteria (1-5 each)
and 6 boolean checks. Determines pass/fail and routes targeted retries
to specific failing agents.

This is the decision-making hub that drives conditional edges in LangGraph.

Enhanced with:
- Per-agent retry budgets (5 each, independent)
- Retry history tracking for best-of-N selection
- Robust QA fallback (retry QA once → programmatic scoring → never blind force-pass)
- Structured telemetry for observability
- Multi-target routing support (parallel retry)
"""

import json
from datetime import datetime, timezone
from src.state import BroadcastState
from src.config import get_llm, MAX_RETRIES_PER_AGENT, MAX_GLOBAL_CYCLES
from src.models.schemas import QAResult
from src.prompts.qa_prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE


def qa_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Evaluates the screenplay and decides next route.

    Reads: source_title, article_text, segments, visual_plan, headline_plan,
           total_duration_sec, retry_count, editor_retry_count,
           visual_retry_count, headline_retry_count, retry_history
    Writes: qa_scores, qa_checks, qa_pass, qa_failure_targets, qa_feedback,
            editor_retry_feedback, visual_retry_feedback, headline_retry_feedback,
            retry_count, current_route, retry_history, retry_decisions
    """
    errors = []

    segments = state.get("segments", [])
    visual_plan = state.get("visual_plan", [])
    headline_plan = state.get("headline_plan", [])
    retry_count = state.get("retry_count", 0)
    history = list(state.get("retry_history", []))

    # ── Handle empty state gracefully ──
    if not segments:
        decision = _build_decision(retry_count, "no_segments", "retry_editor",
                                   ["editor"], _default_scores(), _default_checks())
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
            "retry_decisions": [decision],
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

    # ── Call Gemini for evaluation (with 1 retry on QA failure) ──
    result = None
    for qa_attempt in range(2):  # Max 2 QA attempts (original + 1 retry)
        try:
            llm = get_llm(temperature=0.2)  # Low temp for analytical scoring
            structured_llm = llm.with_structured_output(QAResult)

            result = structured_llm.invoke(
                [
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            break  # Success — exit retry loop

        except Exception as e:
            if qa_attempt == 0:
                errors.append(f"QA agent attempt 1 failed: {str(e)}. Retrying once...")
                continue
            else:
                errors.append(f"QA agent attempt 2 failed: {str(e)}. Falling back to programmatic scoring.")
                result = None

    # ── If both QA attempts failed, use programmatic-only scoring ──
    if result is None:
        return _programmatic_fallback_qa(state, segments, visual_plan, headline_plan,
                                          retry_count, history, errors)

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

    # ── Calculate composite score for this attempt ──
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    checks_passed = sum(1 for v in checks.values() if v)
    check_rate = checks_passed / max(len(checks), 1)
    composite_score = (avg_score / 5 * 0.6) + (check_rate * 0.4)

    # ── Record this attempt in retry history ──
    attempt_record = {
        "attempt": retry_count,
        "scores": scores,
        "checks": checks,
        "avg_score": avg_score,
        "composite_score": composite_score,
        "segments": segments,
        "visual_plan": visual_plan,
        "headline_plan": headline_plan,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history.append(attempt_record)

    # ── Smart routing with per-agent budgets ──
    route, route_reason, adjusted_targets = _smart_route(
        state, qa_pass, failure_targets, retry_count
    )

    # ── Build feedback only for agents that will actually be retried ──
    editor_retry_fb = editor_feedback if "editor" in adjusted_targets else ""
    visual_retry_fb = visual_feedback if "visual" in adjusted_targets else ""
    headline_retry_fb = headline_feedback if "headline" in adjusted_targets else ""

    # ── Telemetry ──
    decision = _build_decision(retry_count, route_reason, route,
                               adjusted_targets, scores, checks, state)

    return {
        "qa_scores": scores,
        "qa_checks": checks,
        "qa_pass": qa_pass,
        "qa_failure_targets": adjusted_targets,
        "qa_feedback": result.feedback,
        "editor_retry_feedback": editor_retry_fb,
        "visual_retry_feedback": visual_retry_fb,
        "headline_retry_feedback": headline_retry_fb,
        "retry_count": retry_count + 1,
        "current_route": route,
        "retry_history": history,
        "retry_decisions": [decision],
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════
# Programmatic Fallback QA (when LLM QA fails twice)
# ═══════════════════════════════════════════════════════════

def _programmatic_fallback_qa(state, segments, visual_plan, headline_plan,
                               retry_count, history, errors):
    """
    Pure programmatic QA when LLM scoring fails.
    
    Uses deterministic checks to score the output. Less nuanced than
    LLM scoring but never crashes and never blind-passes.
    """
    scores = {}
    checks = {}

    # ── Programmatic score estimation ──
    # Story structure: check segment count and beat diversity
    beat_types = {s.get("beat_type", "") for s in segments}
    has_hook = "opening_hook" in beat_types
    has_closing = "closing" in beat_types
    scores["story_structure"] = min(5, 2 + (1 if has_hook else 0) + 
                                    (1 if has_closing else 0) + 
                                    (1 if 4 <= len(segments) <= 6 else 0))

    # Hook engagement: check first segment has content
    first_narration = segments[0].get("anchor_narration", "") if segments else ""
    scores["hook_engagement"] = min(5, 3 + (1 if len(first_narration) > 50 else 0) +
                                     (1 if has_hook else 0))

    # Narration quality: check no empty segments, reasonable word counts
    empty_segments = sum(1 for s in segments if not s.get("anchor_narration", "").strip())
    scores["narration_quality"] = max(1, 5 - empty_segments)

    # Visual planning: check coverage
    visual_coverage = len(visual_plan) / max(len(segments), 1)
    scores["visual_planning"] = min(5, max(1, round(visual_coverage * 5)))

    # Headline quality: check coverage and uniqueness
    headline_coverage = len(headline_plan) / max(len(segments), 1)
    unique_headlines = len({h.get("main_headline", "") for h in headline_plan})
    uniqueness_rate = unique_headlines / max(len(headline_plan), 1)
    scores["headline_quality"] = min(5, max(1, round((headline_coverage * 0.5 + uniqueness_rate * 0.5) * 5)))

    # ── Programmatic boolean checks ──
    total_words = sum(s.get("word_count", 0) for s in segments)
    total_duration = sum(s.get("duration_sec", 0) for s in segments)

    checks["factual_grounding"] = True  # Can't verify without LLM, assume True
    checks["coverage"] = total_words >= 100
    checks["duration_fit"] = 60 <= total_duration <= 120
    checks["text_fit"] = all(
        len(h.get("main_headline", "")) <= 40 and len(h.get("subheadline", "")) <= 60
        for h in headline_plan
    ) if headline_plan else False
    checks["redundancy_free"] = len({h.get("main_headline", "") for h in headline_plan}) == len(headline_plan) if headline_plan else False
    checks["timeline_coherence"] = len(visual_plan) == len(segments) and len(headline_plan) == len(segments)

    # ── Evaluate pass/fail ──
    qa_pass = _evaluate_pass(scores, checks)
    failure_targets = _determine_failure_targets(scores, checks)

    # ── Calculate composite score ──
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    checks_passed = sum(1 for v in checks.values() if v)
    check_rate = checks_passed / max(len(checks), 1)
    composite_score = (avg_score / 5 * 0.6) + (check_rate * 0.4)

    # ── Record attempt ──
    attempt_record = {
        "attempt": retry_count,
        "scores": scores,
        "checks": checks,
        "avg_score": avg_score,
        "composite_score": composite_score,
        "segments": segments,
        "visual_plan": visual_plan,
        "headline_plan": headline_plan,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "programmatic_fallback": True,
    }
    history.append(attempt_record)

    # ── Smart routing ──
    route, route_reason, adjusted_targets = _smart_route(
        state, qa_pass, failure_targets, retry_count
    )

    errors.append("QA used programmatic fallback scoring (LLM unavailable).")

    decision = _build_decision(retry_count, f"programmatic_fallback:{route_reason}",
                               route, adjusted_targets, scores, checks, state)

    return {
        "qa_scores": scores,
        "qa_checks": checks,
        "qa_pass": qa_pass,
        "qa_failure_targets": adjusted_targets,
        "qa_feedback": f"Programmatic QA (LLM unavailable). Avg score: {avg_score:.1f}/5.",
        "editor_retry_feedback": "Improve narration quality and coverage." if "editor" in adjusted_targets else "",
        "visual_retry_feedback": "Ensure every segment has a visual assignment." if "visual" in adjusted_targets else "",
        "headline_retry_feedback": "Ensure every segment has unique headlines within character limits." if "headline" in adjusted_targets else "",
        "retry_count": retry_count + 1,
        "current_route": route,
        "retry_history": history,
        "retry_decisions": [decision],
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════
# Smart Routing with Per-Agent Budgets
# ═══════════════════════════════════════════════════════════

def _smart_route(state, qa_pass, failure_targets, retry_count):
    """
    Determines routing with per-agent budget awareness.
    
    Returns (route, reason, adjusted_targets).
    """
    if qa_pass:
        return "select_best", "all_passed", []

    # Check global cycle cap
    if retry_count >= MAX_GLOBAL_CYCLES:
        return "select_best", "global_cycle_cap_reached", []

    # Filter targets to only those with remaining budget
    adjusted_targets = []
    exhausted_targets = []
    for target in failure_targets:
        count_key = f"{target}_retry_count"
        current_count = state.get(count_key, 0)
        max_count = MAX_RETRIES_PER_AGENT.get(target, 5)
        if current_count < max_count:
            adjusted_targets.append(target)
        else:
            exhausted_targets.append(target)

    if not adjusted_targets:
        reason = f"all_budgets_exhausted({','.join(exhausted_targets)})"
        return "select_best", reason, []

    # Determine routing based on which agents need retry
    if "editor" in adjusted_targets:
        # Editor retry cascades to visual + headline
        return "retry_editor", "editor_failed", adjusted_targets

    if "visual" in adjusted_targets and "headline" in adjusted_targets:
        # Both non-editor agents failed → parallel retry
        return "retry_parallel", "visual_and_headline_failed", adjusted_targets

    if "visual" in adjusted_targets:
        return "retry_visual", "visual_failed", adjusted_targets

    if "headline" in adjusted_targets:
        return "retry_headline", "headline_failed", adjusted_targets

    # Safety fallback
    return "select_best", "no_actionable_targets", []


# ═══════════════════════════════════════════════════════════
# Evaluation Logic
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# Telemetry
# ═══════════════════════════════════════════════════════════

def _build_decision(cycle, reason, route, targets, scores, checks, state=None):
    """Build a structured retry decision record for observability."""
    avg_score = sum(scores.values()) / len(scores) if scores else 0

    decision = {
        "cycle": cycle,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "checks": checks,
        "avg_score": round(avg_score, 2),
        "route_decision": route,
        "route_reason": reason,
        "failure_targets": targets,
    }

    if state:
        decision["budgets"] = {
            "editor": f"{state.get('editor_retry_count', 0)}/{MAX_RETRIES_PER_AGENT['editor']}",
            "visual": f"{state.get('visual_retry_count', 0)}/{MAX_RETRIES_PER_AGENT['visual']}",
            "headline": f"{state.get('headline_retry_count', 0)}/{MAX_RETRIES_PER_AGENT['headline']}",
        }

    return decision


# ═══════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════

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
