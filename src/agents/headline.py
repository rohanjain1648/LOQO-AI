"""
Agent 3b: Headline Generator Agent

Generates per-segment changing headlines, subheadlines, and top tags.
Runs in PARALLEL with the Visual Packaging Agent.
"""

import json
from src.state import BroadcastState
from src.config import get_llm
from src.models.schemas import HeadlineOutput
from src.prompts.headline_prompts import (
    HEADLINE_SYSTEM_PROMPT,
    HEADLINE_USER_TEMPLATE,
    HEADLINE_RETRY_SECTION,
    HEADLINE_NO_RETRY,
)


def headline_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Generates headlines for each narration segment.

    Reads: segments, headline_retry_feedback
    Writes: headline_plan
    """
    errors = []

    segments = state.get("segments", [])

    if not segments:
        errors.append("Headline agent received empty segments. Cannot generate headlines.")
        return {
            "headline_plan": [],
            "headline_retry_feedback": "",
            "errors": errors,
        }

    # ── Build retry section ──
    retry_feedback = state.get("headline_retry_feedback", "")
    if retry_feedback:
        retry_section = HEADLINE_RETRY_SECTION.format(feedback=retry_feedback)
    else:
        retry_section = HEADLINE_NO_RETRY

    # ── Build user prompt ──
    user_prompt = HEADLINE_USER_TEMPLATE.format(
        segments_json=json.dumps(segments, indent=2),
        retry_section=retry_section,
    )

    # ── Call Gemini ──
    try:
        llm = get_llm(temperature=0.7)
        structured_llm = llm.with_structured_output(HeadlineOutput)

        result = structured_llm.invoke(
            [
                {"role": "system", "content": HEADLINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        headline_plan = [ha.model_dump() for ha in result.headline_assignments]

        # ── Validate: every segment must have headlines ──
        planned_ids = {h["segment_id"] for h in headline_plan}
        segment_ids = {s["segment_id"] for s in segments}
        missing = segment_ids - planned_ids

        if missing:
            errors.append(
                f"Headline plan missing segments: {missing}. "
                f"QA will flag this."
            )

        # ── Validate: character limits ──
        for h in headline_plan:
            if len(h.get("main_headline", "")) > 40:
                errors.append(
                    f"Segment {h['segment_id']}: headline '{h['main_headline']}' "
                    f"exceeds 40 char limit ({len(h['main_headline'])} chars)."
                )
            if len(h.get("subheadline", "")) > 60:
                errors.append(
                    f"Segment {h['segment_id']}: subheadline '{h['subheadline']}' "
                    f"exceeds 60 char limit ({len(h['subheadline'])} chars)."
                )

        # ── Validate: no adjacent duplicate headlines ──
        for i in range(1, len(headline_plan)):
            if (headline_plan[i].get("main_headline") ==
                    headline_plan[i - 1].get("main_headline")):
                errors.append(
                    f"Segments {headline_plan[i-1]['segment_id']} and "
                    f"{headline_plan[i]['segment_id']} share the same headline: "
                    f"'{headline_plan[i]['main_headline']}'"
                )

        return {
            "headline_plan": headline_plan,
            "headline_retry_feedback": "",
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Headline agent error: {str(e)}")
        return {
            "headline_plan": [],
            "headline_retry_feedback": "",
            "errors": errors,
        }
