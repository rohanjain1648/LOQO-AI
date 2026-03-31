"""
Best-of-N Selector — picks the highest-scoring attempt from retry history.

Instead of always using the last retry attempt, this selector scores every
attempt using a composite formula and restores the best one to state.

Composite score = 0.6 * (avg_criterion / 5) + 0.4 * (checks_passed / 6)

This prevents the common failure mode where retry #2 was better than #3
but the system shipped the worse version.
"""

from src.state import BroadcastState
from datetime import datetime


def best_of_n_selector(state: BroadcastState) -> dict:
    """
    LangGraph node: Selects the best attempt from retry_history.
    
    If retry_history is empty or has only 1 entry, uses current state.
    Otherwise, calculates composite scores and restores the best attempt.
    
    Reads: retry_history, segments, visual_plan, headline_plan, qa_scores, qa_checks
    Writes: segments, visual_plan, headline_plan, best_attempt_index, errors
    """
    history = state.get("retry_history", [])
    errors = []

    if not history:
        # No history — use current state as-is
        return {
            "best_attempt_index": 0,
            "errors": errors,
        }

    # ── Score every attempt ──
    scored = []
    for i, attempt in enumerate(history):
        scores = attempt.get("scores", {})
        checks = attempt.get("checks", {})

        # Criterion average (0-1 scale)
        score_values = list(scores.values()) if scores else [0]
        avg_criterion = sum(score_values) / len(score_values) / 5.0

        # Boolean check pass rate (0-1 scale)
        check_values = list(checks.values()) if checks else [False]
        checks_passed = sum(1 for v in check_values if v)
        check_rate = checks_passed / max(len(check_values), 1)

        # Composite: 60% criterion quality, 40% constraint compliance
        composite = (avg_criterion * 0.6) + (check_rate * 0.4)

        scored.append({
            "index": i,
            "composite": composite,
            "avg_criterion": avg_criterion * 5,  # back to 1-5 scale for logging
            "check_rate": check_rate,
        })

    # ── Find the best attempt ──
    best = max(scored, key=lambda x: x["composite"])
    best_index = best["index"]
    best_attempt = history[best_index]

    # ── Log the selection decision ──
    if len(history) > 1:
        errors.append(
            f"Best-of-{len(history)} selector: chose attempt #{best_index + 1} "
            f"(composite={best['composite']:.3f}, "
            f"avg={best['avg_criterion']:.1f}/5, "
            f"checks={best['check_rate']:.0%})"
        )

    # ── Check if current state (last attempt) is the best ──
    if best_index == len(history) - 1:
        # Last attempt is best — no state restoration needed
        return {
            "best_attempt_index": best_index,
            "errors": errors,
        }

    # ── Restore the best attempt's outputs to state ──
    errors.append(
        f"Restoring attempt #{best_index + 1} outputs (current attempt #{len(history)} "
        f"scored lower: {scored[-1]['composite']:.3f} vs {best['composite']:.3f})"
    )

    return {
        "segments": best_attempt.get("segments", state.get("segments", [])),
        "visual_plan": best_attempt.get("visual_plan", state.get("visual_plan", [])),
        "headline_plan": best_attempt.get("headline_plan", state.get("headline_plan", [])),
        "qa_scores": best_attempt.get("scores", state.get("qa_scores", {})),
        "qa_checks": best_attempt.get("checks", state.get("qa_checks", {})),
        "best_attempt_index": best_index,
        "errors": errors,
    }
