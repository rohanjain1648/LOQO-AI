"""
Agent 3b: Headline Generator Agent

Generates per-segment changing headlines, subheadlines, and top tags.
Runs in PARALLEL with the Visual Packaging Agent.

Enhanced with:
- Progressive prompt escalation (3 tiers)
- Previous output diff context for retries
- Per-agent retry budget tracking
- Temperature reduction on retries
"""

import json
from src.state import BroadcastState
from src.config import get_llm, get_llm_for_retry, MAX_RETRIES_PER_AGENT
from src.models.schemas import HeadlineOutput
from src.prompts.headline_prompts import (
    HEADLINE_SYSTEM_PROMPT,
    HEADLINE_USER_TEMPLATE,
    HEADLINE_NO_RETRY,
    get_headline_retry_section,
)


def headline_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Generates headlines for each narration segment.

    Reads: segments, headline_retry_feedback, previous_headline_plan,
           headline_retry_count, retry_history
    Writes: headline_plan, previous_headline_plan, headline_retry_count
    """
    errors = []

    segments = state.get("segments", [])
    headline_retry_count = state.get("headline_retry_count", 0)
    max_attempts = MAX_RETRIES_PER_AGENT["headline"]

    if not segments:
        errors.append("Headline agent received empty segments. Cannot generate headlines.")
        return {
            "headline_plan": [],
            "headline_retry_feedback": "",
            "errors": errors,
        }

    # ── Build retry section with progressive escalation ──
    retry_feedback = state.get("headline_retry_feedback", "")
    if retry_feedback:
        previous_headlines = state.get("previous_headline_plan", [])
        previous_headlines_json = json.dumps(previous_headlines, indent=2) if previous_headlines else ""

        # Get best previous from retry history for tier 3
        history = state.get("retry_history", [])
        best_headlines_json = ""
        if history:
            best_idx = max(range(len(history)), key=lambda i: history[i].get("composite_score", 0))
            best_headlines = history[best_idx].get("headline_plan", [])
            best_headlines_json = json.dumps(best_headlines, indent=2) if best_headlines else ""

        retry_section = get_headline_retry_section(
            attempt=headline_retry_count + 1,
            max_attempts=max_attempts,
            feedback=retry_feedback,
            previous_headlines_json=previous_headlines_json,
            best_previous_headlines_json=best_headlines_json,
            specific_fix_list=retry_feedback,
        )
    else:
        retry_section = HEADLINE_NO_RETRY

    # ── Build user prompt ──
    user_prompt = HEADLINE_USER_TEMPLATE.format(
        segments_json=json.dumps(segments, indent=2),
        retry_section=retry_section,
    )

    # ── Call Gemini with progressive temperature ──
    try:
        if retry_feedback:
            llm = get_llm_for_retry("headline", headline_retry_count)
        else:
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
                f"Post-validation will flag this."
            )

        # ── Validate: character limits (soft check, post-validator auto-fixes) ──
        for h in headline_plan:
            if len(h.get("main_headline", "")) > 40:
                errors.append(
                    f"Segment {h['segment_id']}: headline exceeds 40 chars. "
                    f"Post-validation will auto-fix."
                )
            if len(h.get("subheadline", "")) > 60:
                errors.append(
                    f"Segment {h['segment_id']}: subheadline exceeds 60 chars. "
                    f"Post-validation will auto-fix."
                )

        # ── Validate: no adjacent duplicate headlines ──
        for i in range(1, len(headline_plan)):
            if (headline_plan[i].get("main_headline") ==
                    headline_plan[i - 1].get("main_headline")):
                errors.append(
                    f"Segments {headline_plan[i-1]['segment_id']} and "
                    f"{headline_plan[i]['segment_id']} share the same headline."
                )

        return {
            "headline_plan": headline_plan,
            "previous_headline_plan": state.get("headline_plan", []),  # Save current as previous
            "headline_retry_feedback": "",
            "headline_retry_count": headline_retry_count + (1 if retry_feedback else 0),
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Headline agent error: {str(e)}")
        return {
            "headline_plan": [],
            "headline_retry_feedback": "",
            "headline_retry_count": headline_retry_count + (1 if retry_feedback else 0),
            "errors": errors,
        }
