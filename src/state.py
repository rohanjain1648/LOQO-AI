"""
Central state schema for the LangGraph broadcast screenplay pipeline.

All agents read from and write to this shared state. The `errors` field uses
operator.add as a reducer so that parallel agents (Visual + Headline) can
safely append errors without overwriting each other.

Retry system fields track per-agent budgets, attempt history for best-of-N
selection, and structured telemetry for observability.
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

    # ═══ RETRY HISTORY (best-of-N selection) ═══
    retry_history: list[dict]           # [{attempt, scores, checks, segments,
                                        #   visual_plan, headline_plan, avg_score,
                                        #   composite_score}, ...]
    best_attempt_index: int             # Index into retry_history of the best attempt

    # ═══ PREVIOUS OUTPUTS (fed back to agents on retry for diff context) ═══
    previous_segments: list[dict]       # Editor's last output
    previous_visual_plan: list[dict]    # Visual agent's last output
    previous_headline_plan: list[dict]  # Headline agent's last output

    # ═══ PROGRAMMATIC VALIDATION ═══
    pre_validation_errors: list[dict]   # [{field, issue, severity, auto_fixable}]
    post_validation_errors: list[dict]  # [{field, issue, severity, auto_fixable}]

    # ═══ CONTROL FLOW ═══
    retry_count: int                    # Global cycle counter (0-based)
    current_route: str
    errors: Annotated[list[str], operator.add]  # Parallel-safe via reducer

    # ═══ PER-AGENT RETRY BUDGETS ═══
    editor_retry_count: int             # 0-5 (independent budget)
    visual_retry_count: int             # 0-5
    headline_retry_count: int           # 0-5

    # ═══ TELEMETRY ═══
    retry_decisions: Annotated[list[dict], operator.add]  # [{cycle, decision, reason, ...}]

    # ═══ FINAL OUTPUT ═══
    final_json: dict
    final_screenplay: str
