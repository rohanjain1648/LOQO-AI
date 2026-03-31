"""
Agent 2: News Editor Agent

Takes article text and produces broadcast-style narration broken into
timed segments. Uses Gemini with structured output for reliable parsing.

Enhanced with:
- Progressive prompt escalation (3 tiers)
- Previous output diff context for retries
- Per-agent retry budget tracking
- Temperature reduction on retries
"""

import json
from src.state import BroadcastState
from src.config import get_llm, get_llm_for_retry, MAX_RETRIES_PER_AGENT
from src.models.schemas import EditorOutput
from src.prompts.editor_prompts import (
    EDITOR_SYSTEM_PROMPT,
    EDITOR_USER_TEMPLATE,
    EDITOR_NO_RETRY,
    get_editor_retry_section,
)


def editor_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Generates narration segments from article text.

    Reads: article_text, source_title, extraction_metadata, editor_retry_feedback,
           previous_segments, editor_retry_count, retry_history
    Writes: story_beats, segments, total_duration_sec, previous_segments,
            editor_retry_count
    """
    errors = []

    editor_retry_count = state.get("editor_retry_count", 0)
    max_attempts = MAX_RETRIES_PER_AGENT["editor"]

    # ── Build retry section with progressive escalation ──
    retry_feedback = state.get("editor_retry_feedback", "")
    if retry_feedback:
        # Get previous output context for tier 2+
        previous_segments = state.get("previous_segments", [])
        previous_segments_json = json.dumps(previous_segments, indent=2) if previous_segments else ""

        # Get best previous from retry history for tier 3
        history = state.get("retry_history", [])
        best_segments_json = ""
        score_history = ""
        prev_avg = 0.0

        if history:
            scores_list = [h.get("avg_score", 0) for h in history]
            score_history = ", ".join(f"{s:.1f}/5" for s in scores_list)
            prev_avg = scores_list[-1] if scores_list else 0

            # Find best attempt
            best_idx = max(range(len(history)), key=lambda i: history[i].get("composite_score", 0))
            best_segments = history[best_idx].get("segments", [])
            best_segments_json = json.dumps(best_segments, indent=2) if best_segments else ""

        retry_section = get_editor_retry_section(
            attempt=editor_retry_count + 1,
            max_attempts=max_attempts,
            feedback=retry_feedback,
            previous_segments_json=previous_segments_json,
            prev_avg=prev_avg,
            score_history=score_history,
            best_previous_segments_json=best_segments_json,
            specific_fix_list=retry_feedback,
        )
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

    # ── Call Gemini with progressive temperature ──
    try:
        if retry_feedback:
            llm = get_llm_for_retry("editor", editor_retry_count)
        else:
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
                    f"Validators will attempt auto-fix."
                )

        # ── Convert to dicts for state ──
        story_beats = [beat.model_dump() for beat in result.story_beats]
        segments = [seg.model_dump() for seg in result.segments]

        return {
            "story_beats": story_beats,
            "segments": segments,
            "total_duration_sec": total_duration,
            "previous_segments": state.get("segments", []),  # Save current as previous
            "editor_retry_feedback": "",  # Clear retry feedback after processing
            "editor_retry_count": editor_retry_count + (1 if retry_feedback else 0),
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
            "editor_retry_count": editor_retry_count + (1 if retry_feedback else 0),
            "errors": errors,
        }
