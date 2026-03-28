"""
Output formatter — produces both structured JSON and human-readable screenplay.

This is the final LangGraph node that assembles all agent outputs into
the two required output formats.
"""

import json
from src.state import BroadcastState
from src.output.templates import (
    SCREENPLAY_HEADER,
    SEGMENT_TEMPLATE,
    SOURCE_IMAGE_LINE,
    AI_VISUAL_LINE,
    NO_IMAGE_LINE,
    QA_SCORES_TEMPLATE,
    make_stars,
)


def format_final_output(state: BroadcastState) -> dict:
    """
    LangGraph node: Assembles final output formats.

    Reads: all state fields
    Writes: final_json, final_screenplay
    """
    segments = state.get("segments", [])
    visual_plan = state.get("visual_plan", [])
    headline_plan = state.get("headline_plan", [])

    # ── Build lookup maps for visual and headline data ──
    visual_map = {v["segment_id"]: v for v in visual_plan}
    headline_map = {h["segment_id"]: h for h in headline_plan}

    # ── Assemble merged segments ──
    merged_segments = []
    for seg in segments:
        sid = seg["segment_id"]
        vis = visual_map.get(sid, {})
        hdl = headline_map.get(sid, {})

        merged = {
            "segment_id": sid,
            "start_time": seg.get("start_time", "00:00"),
            "end_time": seg.get("end_time", "00:00"),
            "layout": vis.get("layout", "anchor_only"),
            "anchor_narration": seg.get("anchor_narration", ""),
            "main_headline": hdl.get("main_headline", ""),
            "subheadline": hdl.get("subheadline", ""),
            "top_tag": hdl.get("top_tag", "LATEST"),
            "left_panel": vis.get("left_panel", "AI anchor in studio"),
            "right_panel": vis.get("right_panel", ""),
            "source_image_url": vis.get("source_image_url"),
            "ai_support_visual_prompt": vis.get("ai_support_visual_prompt"),
            "transition": vis.get("transition", "cut"),
        }
        merged_segments.append(merged)

    # ── Build structured JSON ──
    final_json = {
        "article_url": state.get("article_url", ""),
        "source_title": state.get("source_title", ""),
        "video_duration_sec": state.get("total_duration_sec", 0),
        "segments": merged_segments,
    }

    # ── Build human-readable screenplay ──
    final_screenplay = _build_screenplay(state, merged_segments)

    return {
        "final_json": final_json,
        "final_screenplay": final_screenplay,
    }


def _build_screenplay(state: dict, merged_segments: list) -> str:
    """Build the human-readable screenplay text."""
    parts = []

    # ── Header ──
    parts.append(SCREENPLAY_HEADER.format(
        title=state.get("source_title", "Untitled"),
        url=state.get("article_url", ""),
        duration=state.get("total_duration_sec", 0),
        segment_count=len(merged_segments),
    ))

    # ── Segments ──
    for seg in merged_segments:
        # Determine image line
        if seg.get("source_image_url"):
            image_line = SOURCE_IMAGE_LINE.format(url=seg["source_image_url"])
        elif seg.get("ai_support_visual_prompt"):
            image_line = AI_VISUAL_LINE.format(prompt=seg["ai_support_visual_prompt"])
        else:
            image_line = NO_IMAGE_LINE

        parts.append(SEGMENT_TEMPLATE.format(
            segment_id=seg["segment_id"],
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            top_tag=seg.get("top_tag", "LATEST"),
            main_headline=seg.get("main_headline", ""),
            subheadline=seg.get("subheadline", ""),
            layout=seg.get("layout", "anchor_only"),
            left_panel=seg.get("left_panel", "AI anchor in studio"),
            right_panel=seg.get("right_panel", ""),
            image_line=image_line,
            narration=seg.get("anchor_narration", ""),
            transition=seg.get("transition", "cut"),
        ))

    # ── QA Scores ──
    qa_scores = state.get("qa_scores", {})
    if qa_scores:
        avg_score = round(sum(qa_scores.values()) / len(qa_scores), 1) if qa_scores else 0
        pass_status = "✅ PASS" if state.get("qa_pass", False) else "❌ FAIL"

        parts.append(QA_SCORES_TEMPLATE.format(
            stars_structure=make_stars(qa_scores.get("story_structure", 0)),
            score_structure=qa_scores.get("story_structure", 0),
            stars_hook=make_stars(qa_scores.get("hook_engagement", 0)),
            score_hook=qa_scores.get("hook_engagement", 0),
            stars_narration=make_stars(qa_scores.get("narration_quality", 0)),
            score_narration=qa_scores.get("narration_quality", 0),
            stars_visual=make_stars(qa_scores.get("visual_planning", 0)),
            score_visual=qa_scores.get("visual_planning", 0),
            stars_headline=make_stars(qa_scores.get("headline_quality", 0)),
            score_headline=qa_scores.get("headline_quality", 0),
            avg_score=avg_score,
            pass_status=pass_status,
            retry_count=state.get("retry_count", 0),
        ))

    return "\n".join(parts)
