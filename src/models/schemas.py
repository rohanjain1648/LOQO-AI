"""
Pydantic models for structured LLM output and validation.

These models are used with Gemini's `with_structured_output()` method
to ensure type-safe, validated responses from every agent.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


# ═══════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════

class BeatType(str, Enum):
    OPENING_HOOK = "opening_hook"
    KEY_DETAILS = "key_details"
    IMPACT = "impact"
    RESPONSE = "response"
    CONTEXT = "context"
    CLOSING = "closing"


class TopTag(str, Enum):
    BREAKING = "BREAKING"
    LIVE = "LIVE"
    DEVELOPING = "DEVELOPING"
    UPDATE = "UPDATE"
    LATEST = "LATEST"
    EXCLUSIVE = "EXCLUSIVE"


class LayoutType(str, Enum):
    ANCHOR_SOURCE = "anchor_left + source_visual_right"
    ANCHOR_AI = "anchor_left + ai_support_visual_right"
    FULLSCREEN = "fullscreen_visual"
    ANCHOR_ONLY = "anchor_only"


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "crossfade"
    SLIDE = "slide"
    FADE_OUT = "fade_out"


# ═══════════════════════════════════════════════════════════
# Agent 2: Editor Output Models
# ═══════════════════════════════════════════════════════════

class StoryBeat(BaseModel):
    """A single story beat identified from the article."""
    beat_id: int = Field(description="Sequential beat number starting from 1")
    beat_type: str = Field(description="Type: opening_hook, key_details, impact, response, context, or closing")
    summary: str = Field(description="1-2 sentence summary of what this beat covers from the article")


class NarrationSegment(BaseModel):
    """A single narration segment with timing information."""
    segment_id: int = Field(description="Sequential segment number starting from 1")
    start_time: str = Field(description="Start time in MM:SS format, e.g. '00:00'")
    end_time: str = Field(description="End time in MM:SS format, e.g. '00:12'")
    anchor_narration: str = Field(description="The anchor narration text for this segment, written in broadcast style")
    beat_type: str = Field(description="Which story beat this segment corresponds to")
    word_count: int = Field(description="Number of words in the narration")
    duration_sec: int = Field(description="Duration in seconds, calculated from word count at 150 WPM")


class EditorOutput(BaseModel):
    """Complete output from the News Editor Agent."""
    story_beats: List[StoryBeat] = Field(description="4-6 identified story beats from the article")
    segments: List[NarrationSegment] = Field(description="Timed narration segments")
    total_duration_sec: int = Field(description="Total duration in seconds, must be between 60-120")


# ═══════════════════════════════════════════════════════════
# Agent 3a: Visual Packaging Output Models
# ═══════════════════════════════════════════════════════════

class VisualAssignment(BaseModel):
    """Visual plan for a single segment."""
    segment_id: int = Field(description="Matching segment ID from narration")
    layout: str = Field(description="One of: 'anchor_left + source_visual_right', 'anchor_left + ai_support_visual_right', 'fullscreen_visual', 'anchor_only'")
    left_panel: str = Field(description="Description of left panel content, typically 'AI anchor in studio'")
    right_panel: str = Field(description="Description of right panel content")
    source_image_url: Optional[str] = Field(default=None, description="URL of source article image if used, null if AI visual")
    ai_support_visual_prompt: Optional[str] = Field(default=None, description="AI image generation prompt if no source image, null if source image used")
    transition: str = Field(description="Transition type: 'cut', 'crossfade', 'slide', or 'fade_out'")


class VisualOutput(BaseModel):
    """Complete output from the Visual Packaging Agent."""
    visual_assignments: List[VisualAssignment] = Field(description="Visual plan for each segment")


# ═══════════════════════════════════════════════════════════
# Agent 3b: Headline Generator Output Models
# ═══════════════════════════════════════════════════════════

class HeadlineAssignment(BaseModel):
    """Headlines for a single segment."""
    segment_id: int = Field(description="Matching segment ID from narration")
    main_headline: str = Field(description="Main headline, max 40 characters, punchy broadcast style")
    subheadline: str = Field(description="Subheadline, max 60 characters, adds context")
    top_tag: str = Field(description="One of: BREAKING, LIVE, DEVELOPING, UPDATE, LATEST, EXCLUSIVE")


class HeadlineOutput(BaseModel):
    """Complete output from the Headline Generator Agent."""
    headline_assignments: List[HeadlineAssignment] = Field(description="Headlines for each segment")


# ═══════════════════════════════════════════════════════════
# Agent 4: QA / Evaluation Output Models
# ═══════════════════════════════════════════════════════════

class QAScores(BaseModel):
    """Scores for the 5 main evaluation criteria (1-5 each)."""
    story_structure: int = Field(ge=1, le=5, description="Story structure and flow score")
    hook_engagement: int = Field(ge=1, le=5, description="Hook and engagement score")
    narration_quality: int = Field(ge=1, le=5, description="Narration quality score")
    visual_planning: int = Field(ge=1, le=5, description="Visual planning and placement score")
    headline_quality: int = Field(ge=1, le=5, description="Headline and subheadline quality score")


class QAChecks(BaseModel):
    """Boolean checks for additional quality criteria."""
    factual_grounding: bool = Field(description="No invented claims beyond source article")
    coverage: bool = Field(description="Major facts from article are included")
    duration_fit: bool = Field(description="Total narration between 60-120 seconds")
    text_fit: bool = Field(description="Headlines ≤40 chars, subheadlines ≤60 chars")
    redundancy_free: bool = Field(description="No repeated facts/headlines/visuals")
    timeline_coherence: bool = Field(description="Narration, visuals, overlays align per segment")


class QAResult(BaseModel):
    """Complete output from the QA Evaluation Agent."""
    scores: QAScores
    checks: QAChecks
    overall_pass: bool = Field(description="True if all criteria met: no score below 3, avg >= 4, all checks pass")
    failure_targets: List[str] = Field(description="List of agents that need retry: 'editor', 'visual', 'headline'")
    feedback: str = Field(description="Overall feedback summary")
    editor_feedback: Optional[str] = Field(default=None, description="Specific actionable feedback for Editor agent")
    visual_feedback: Optional[str] = Field(default=None, description="Specific actionable feedback for Visual agent")
    headline_feedback: Optional[str] = Field(default=None, description="Specific actionable feedback for Headline agent")


# ═══════════════════════════════════════════════════════════
# Final Output Models
# ═══════════════════════════════════════════════════════════

class FinalSegment(BaseModel):
    """A complete segment combining narration, visuals, and headlines."""
    segment_id: int
    start_time: str
    end_time: str
    layout: str
    anchor_narration: str
    main_headline: str
    subheadline: str
    top_tag: str
    left_panel: str
    right_panel: str
    source_image_url: Optional[str] = None
    ai_support_visual_prompt: Optional[str] = None
    transition: str


class BroadcastScreenplay(BaseModel):
    """The complete broadcast screenplay output."""
    article_url: str
    source_title: str
    video_duration_sec: int
    segments: List[FinalSegment]
