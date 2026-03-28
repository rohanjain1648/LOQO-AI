"""
Central state schema for the LangGraph broadcast screenplay pipeline.

All agents read from and write to this shared state. The `errors` field uses
operator.add as a reducer so that parallel agents (Visual + Headline) can
safely append errors without overwriting each other.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator


class BroadcastState(TypedDict):
    """The single shared state object flowing through the entire LangGraph pipeline."""

    # ═══ INPUT ═══
    article_url: str

    # ═══ AGENT 1: EXTRACTION OUTPUT ═══
    source_title: str
    article_text: str
    source_images: list[dict]           # [{url, alt, context}, ...]
    extraction_metadata: dict           # {author, date, source_name}

    # ═══ AGENT 2: EDITOR OUTPUT ═══
    story_beats: list[dict]
    segments: list[dict]                # [{segment_id, start_time, end_time,
                                        #   anchor_narration, beat_type,
                                        #   word_count, duration_sec}, ...]
    total_duration_sec: int

    # ═══ AGENT 3a: VISUAL PACKAGING OUTPUT ═══
    visual_plan: list[dict]             # [{segment_id, layout, left_panel,
                                        #   right_panel, source_image_url,
                                        #   ai_support_visual_prompt, transition}]

    # ═══ AGENT 3b: HEADLINE GENERATOR OUTPUT ═══
    headline_plan: list[dict]           # [{segment_id, main_headline,
                                        #   subheadline, top_tag}]

    # ═══ AGENT 4: QA / EVALUATION OUTPUT ═══
    qa_scores: dict                     # {story_structure: 4, ...}
    qa_checks: dict                     # {factual_grounding: True, ...}
    qa_pass: bool
    qa_failure_targets: list[str]       # ["editor", "visual", "headline"]
    qa_feedback: str

    # ═══ RETRY FEEDBACK (per-agent) ═══
    editor_retry_feedback: str
    visual_retry_feedback: str
    headline_retry_feedback: str

    # ═══ CONTROL FLOW ═══
    retry_count: int                    # Current iteration (0-3)
    current_route: str
    errors: Annotated[list[str], operator.add]  # Parallel-safe via reducer

    # ═══ FINAL OUTPUT ═══
    final_json: dict
    final_screenplay: str
