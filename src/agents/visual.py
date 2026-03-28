"""
Agent 3a: Visual Packaging Agent

Takes narration segments and source images, then plans per-segment
visuals with layouts, image assignments, AI prompts, and transitions.
Runs in PARALLEL with the Headline Agent.
"""

import json
from src.state import BroadcastState
from src.config import get_llm
from src.models.schemas import VisualOutput
from src.prompts.visual_prompts import (
    VISUAL_SYSTEM_PROMPT,
    VISUAL_USER_TEMPLATE,
    VISUAL_RETRY_SECTION,
    VISUAL_NO_RETRY,
)


def visual_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Plans visuals for each narration segment.

    Reads: segments, source_images, visual_retry_feedback
    Writes: visual_plan
    """
    errors = []

    segments = state.get("segments", [])
    source_images = state.get("source_images", [])

    if not segments:
        errors.append("Visual agent received empty segments. Cannot plan visuals.")
        return {
            "visual_plan": [],
            "visual_retry_feedback": "",
            "errors": errors,
        }

    # ── Build retry section ──
    retry_feedback = state.get("visual_retry_feedback", "")
    if retry_feedback:
        retry_section = VISUAL_RETRY_SECTION.format(feedback=retry_feedback)
    else:
        retry_section = VISUAL_NO_RETRY

    # ── Build user prompt ──
    user_prompt = VISUAL_USER_TEMPLATE.format(
        segments_json=json.dumps(segments, indent=2),
        source_images_json=json.dumps(source_images, indent=2) if source_images else "No source images available — use AI-generated visuals for all segments.",
        retry_section=retry_section,
    )

    # ── Call Gemini ──
    try:
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
                f"QA will flag this."
            )

        # ── Validate: mutual exclusivity of source_image_url and ai_support_visual_prompt ──
        for v in visual_plan:
            if v.get("source_image_url") and v.get("ai_support_visual_prompt"):
                errors.append(
                    f"Segment {v['segment_id']}: has both source_image_url and ai_support_visual_prompt. "
                    f"Only one should be set."
                )

        # ── Validate: last segment has fade_out ──
        if visual_plan and visual_plan[-1].get("transition") != "fade_out":
            errors.append("Last segment should use 'fade_out' transition.")

        return {
            "visual_plan": visual_plan,
            "visual_retry_feedback": "",
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Visual agent error: {str(e)}")
        return {
            "visual_plan": [],
            "visual_retry_feedback": "",
            "errors": errors,
        }
