"""
Agent 3a: Visual Packaging Agent

Takes narration segments and source images, then plans per-segment
visuals with layouts, image assignments, AI prompts, and transitions.
Runs in PARALLEL with the Headline Agent.

Enhanced with:
- Progressive prompt escalation (3 tiers)
- Previous output diff context for retries
- Per-agent retry budget tracking
- Temperature reduction on retries
"""

import json
from src.state import BroadcastState
from src.config import get_llm, get_llm_for_retry, MAX_RETRIES_PER_AGENT
from src.models.schemas import VisualOutput
from src.prompts.visual_prompts import (
    VISUAL_SYSTEM_PROMPT,
    VISUAL_USER_TEMPLATE,
    VISUAL_NO_RETRY,
    get_visual_retry_section,
)


def visual_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Plans visuals for each narration segment.

    Reads: segments, source_images, visual_retry_feedback,
           previous_visual_plan, visual_retry_count, retry_history
    Writes: visual_plan, previous_visual_plan, visual_retry_count
    """
    errors = []

    segments = state.get("segments", [])
    source_images = state.get("source_images", [])
    visual_retry_count = state.get("visual_retry_count", 0)
    max_attempts = MAX_RETRIES_PER_AGENT["visual"]

    if not segments:
        errors.append("Visual agent received empty segments. Cannot plan visuals.")
        return {
            "visual_plan": [],
            "visual_retry_feedback": "",
            "errors": errors,
        }

    # ── Build retry section with progressive escalation ──
    retry_feedback = state.get("visual_retry_feedback", "")
    if retry_feedback:
        previous_visual = state.get("previous_visual_plan", [])
        previous_visual_json = json.dumps(previous_visual, indent=2) if previous_visual else ""

        # Get best previous from retry history for tier 3
        history = state.get("retry_history", [])
        best_visual_json = ""
        if history:
            best_idx = max(range(len(history)), key=lambda i: history[i].get("composite_score", 0))
            best_visual = history[best_idx].get("visual_plan", [])
            best_visual_json = json.dumps(best_visual, indent=2) if best_visual else ""

        retry_section = get_visual_retry_section(
            attempt=visual_retry_count + 1,
            max_attempts=max_attempts,
            feedback=retry_feedback,
            previous_visual_json=previous_visual_json,
            best_previous_visual_json=best_visual_json,
            specific_fix_list=retry_feedback,
        )
    else:
        retry_section = VISUAL_NO_RETRY

    # ── Build user prompt ──
    user_prompt = VISUAL_USER_TEMPLATE.format(
        segments_json=json.dumps(segments, indent=2),
        source_images_json=json.dumps(source_images, indent=2) if source_images else "No source images available — use AI-generated visuals for all segments.",
        retry_section=retry_section,
    )

    # ── Call Gemini with progressive temperature ──
    try:
        if retry_feedback:
            llm = get_llm_for_retry("visual", visual_retry_count)
        else:
            llm = get_llm(temperature=0.6)

        structured_llm = llm.with_structured_output(VisualOutput)

        result = structured_llm.invoke(
            [
                {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        visual_plan = [va.model_dump() for va in result.visual_assignments]

        # ── Validate: every segment must have a visual ──
        planned_ids = {v["segment_id"] for v in visual_plan}
        segment_ids = {s["segment_id"] for s in segments}
        missing = segment_ids - planned_ids

        if missing:
            errors.append(
                f"Visual plan missing segments: {missing}. "
                f"Post-validation will flag this."
            )

        # ── Validate: mutual exclusivity ──
        for v in visual_plan:
            if v.get("source_image_url") and v.get("ai_support_visual_prompt"):
                errors.append(
                    f"Segment {v['segment_id']}: has both source_image_url and "
                    f"ai_support_visual_prompt. Post-validation will auto-fix."
                )

        # ── Validate: last segment has fade_out ──
        if visual_plan and visual_plan[-1].get("transition") != "fade_out":
            errors.append("Last segment should use 'fade_out' transition. Post-validation will auto-fix.")

        return {
            "visual_plan": visual_plan,
            "previous_visual_plan": state.get("visual_plan", []),  # Save current as previous
            "visual_retry_feedback": "",
            "visual_retry_count": visual_retry_count + (1 if retry_feedback else 0),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Visual agent error: {str(e)}")
        return {
            "visual_plan": [],
            "visual_retry_feedback": "",
            "visual_retry_count": visual_retry_count + (1 if retry_feedback else 0),
            "errors": errors,
        }
