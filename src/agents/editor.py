"""
Agent 2: News Editor Agent

Takes article text and produces broadcast-style narration broken into
timed segments. Uses Gemini with structured output for reliable parsing.
"""

import json
from src.state import BroadcastState
from src.config import get_llm
from src.models.schemas import EditorOutput
from src.prompts.editor_prompts import (
    EDITOR_SYSTEM_PROMPT,
    EDITOR_USER_TEMPLATE,
    EDITOR_RETRY_SECTION,
    EDITOR_NO_RETRY,
)


def editor_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Generates narration segments from article text.

    Reads: article_text, source_title, extraction_metadata, editor_retry_feedback
    Writes: story_beats, segments, total_duration_sec
    """
    errors = []

    # ── Build retry section if this is a retry ──
    retry_feedback = state.get("editor_retry_feedback", "")
    if retry_feedback:
        retry_section = EDITOR_RETRY_SECTION.format(feedback=retry_feedback)
    else:
        retry_section = EDITOR_NO_RETRY

    # ── Build the user prompt ──
    metadata = state.get("extraction_metadata", {})
    user_prompt = EDITOR_USER_TEMPLATE.format(
        title=state.get("source_title", "Untitled"),
        source_name=metadata.get("source_name", "Unknown"),
        date=metadata.get("date", "Unknown"),
        article_text=state.get("article_text", ""),
        retry_section=retry_section,
    )

    # ── Call Gemini with structured output ──
    try:
        llm = get_llm(temperature=0.7)
        structured_llm = llm.with_structured_output(EditorOutput)

        result = structured_llm.invoke(
            [
                {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        # ── Validate duration bounds ──
        total_duration = result.total_duration_sec

        if total_duration < 60 or total_duration > 120:
            # Recalculate from actual word counts
            total_duration = sum(
                seg.duration_sec for seg in result.segments
            )
            if total_duration < 60 or total_duration > 120:
                errors.append(
                    f"Duration {total_duration}s is outside 60-120s range. "
                    f"QA will flag this for retry."
                )

        # ── Convert to dicts for state ──
        story_beats = [beat.model_dump() for beat in result.story_beats]
        segments = [seg.model_dump() for seg in result.segments]

        return {
            "story_beats": story_beats,
            "segments": segments,
            "total_duration_sec": total_duration,
            "editor_retry_feedback": "",  # Clear retry feedback after processing
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Editor agent error: {str(e)}")

        # Return minimal fallback structure
        return {
            "story_beats": [],
            "segments": [],
            "total_duration_sec": 0,
            "editor_retry_feedback": "",
            "errors": errors,
        }
