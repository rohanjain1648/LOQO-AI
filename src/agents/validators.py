"""
Programmatic Validators — deterministic pre/post validation + auto-fix.

These validators catch issues that don't need an LLM call:
- Duration math, character limits, segment counts, timing continuity
- Auto-fixes trivial issues inline (truncation, recalculation)
- Returns structured results indicating what needs LLM retry vs. what was auto-fixed

Two phases:
  Phase A (pre_validate_editor): runs after Editor, before fan-out
  Phase B (post_validate_parallel): runs after Visual + Headline, before QA
"""

from src.state import BroadcastState
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# Validation Result
# ═══════════════════════════════════════════════════════════

class ValidationError:
    """A single validation failure."""
    def __init__(self, field: str, issue: str, severity: str = "error", auto_fixable: bool = False):
        self.field = field
        self.issue = issue
        self.severity = severity  # "error" | "warning"
        self.auto_fixable = auto_fixable

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "issue": self.issue,
            "severity": self.severity,
            "auto_fixable": self.auto_fixable,
        }


class ValidationResult:
    """Aggregated validation outcome."""
    def __init__(self):
        self.errors: list[ValidationError] = []
        self.auto_fixed: list[str] = []
        self.needs_llm_retry: list[str] = []  # which agents need LLM retry

    @property
    def is_valid(self) -> bool:
        return len([e for e in self.errors if e.severity == "error"]) == 0

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "auto_fixed": self.auto_fixed,
            "needs_llm_retry": self.needs_llm_retry,
        }


# ═══════════════════════════════════════════════════════════
# Phase A: Post-Editor Pre-Validation
# ═══════════════════════════════════════════════════════════

def pre_validate_editor(state: BroadcastState) -> dict:
    """
    LangGraph node: Validates + auto-fixes editor output before fan-out.
    
    Checks:
    ✓ Segment count in range [4, 6]
    ✓ Total word count in range [150, 300]
    ✓ Total duration in range [60, 120] seconds
    ✓ Timing continuity (no gaps/overlaps)
    ✓ First segment starts at "00:00"
    ✓ No empty narration text
    ✓ Beat type coverage (opening_hook + closing present)
    
    Auto-fixes:
    - Recalculates duration from word counts if mismatched
    - Fixes timing gaps by recalculating sequential times
    """
    result = ValidationResult()
    segments = list(state.get("segments", []))
    errors_log = []

    if not segments:
        result.errors.append(ValidationError(
            "segments", "No segments generated", "error", False
        ))
        result.needs_llm_retry.append("editor")
        return {
            "pre_validation_errors": [e.to_dict() for e in result.errors],
            "errors": [f"Pre-validation: {result.errors[0].issue}"],
        }

    # ── Check 1: Segment count ──
    if len(segments) < 4:
        result.errors.append(ValidationError(
            "segments", f"Only {len(segments)} segments (minimum 4)", "error", False
        ))
        result.needs_llm_retry.append("editor")
    elif len(segments) > 6:
        result.errors.append(ValidationError(
            "segments", f"{len(segments)} segments (maximum 6)", "warning", False
        ))

    # ── Check 2: Total word count ──
    total_words = sum(
        s.get("word_count", len(s.get("anchor_narration", "").split()))
        for s in segments
    )
    if total_words < 150:
        result.errors.append(ValidationError(
            "word_count", f"Total {total_words} words (minimum 150)", "error", False
        ))
        result.needs_llm_retry.append("editor")
    elif total_words > 300:
        result.errors.append(ValidationError(
            "word_count", f"Total {total_words} words (maximum 300)", "warning", False
        ))

    # ── Check 3: No empty narration ──
    for seg in segments:
        narration = seg.get("anchor_narration", "").strip()
        if not narration:
            result.errors.append(ValidationError(
                f"segment_{seg.get('segment_id', '?')}_narration",
                f"Segment {seg.get('segment_id', '?')} has empty narration",
                "error", False
            ))
            if "editor" not in result.needs_llm_retry:
                result.needs_llm_retry.append("editor")

    # ── Auto-fix: Recalculate word counts and duration from actual narration ──
    recalculated = False
    for seg in segments:
        narration = seg.get("anchor_narration", "")
        actual_words = len(narration.split())
        if seg.get("word_count", 0) != actual_words:
            seg["word_count"] = actual_words
            recalculated = True
        actual_duration = max(1, round(actual_words / 2.5))
        if seg.get("duration_sec", 0) != actual_duration:
            seg["duration_sec"] = actual_duration
            recalculated = True

    if recalculated:
        result.auto_fixed.append("Recalculated word counts and durations from actual narration text")

    # ── Auto-fix: Recalculate sequential timing ──
    current_sec = 0
    timing_fixed = False
    for seg in segments:
        dur = seg.get("duration_sec", 10)
        expected_start = _seconds_to_mmss(current_sec)
        expected_end = _seconds_to_mmss(current_sec + dur)
        if seg.get("start_time") != expected_start or seg.get("end_time") != expected_end:
            seg["start_time"] = expected_start
            seg["end_time"] = expected_end
            timing_fixed = True
        current_sec += dur

    if timing_fixed:
        result.auto_fixed.append("Recalculated sequential timing to eliminate gaps/overlaps")

    # ── Auto-fix: Recalculate total duration ──
    total_duration = sum(s.get("duration_sec", 0) for s in segments)

    # ── Check 4: Duration range ──
    if total_duration < 60:
        result.errors.append(ValidationError(
            "duration", f"Total {total_duration}s (minimum 60s)", "error", False
        ))
        if "editor" not in result.needs_llm_retry:
            result.needs_llm_retry.append("editor")
    elif total_duration > 120:
        result.errors.append(ValidationError(
            "duration", f"Total {total_duration}s (maximum 120s)", "warning", False
        ))

    # ── Check 5: Beat type coverage ──
    beat_types = {s.get("beat_type", "") for s in segments}
    if "opening_hook" not in beat_types:
        result.errors.append(ValidationError(
            "beat_types", "Missing 'opening_hook' beat type", "warning", False
        ))
    if "closing" not in beat_types:
        result.errors.append(ValidationError(
            "beat_types", "Missing 'closing' beat type", "warning", False
        ))

    # ── Build state update ──
    update = {
        "segments": segments,
        "total_duration_sec": total_duration,
        "pre_validation_errors": [e.to_dict() for e in result.errors],
        "errors": errors_log,
    }

    if result.auto_fixed:
        errors_log.append(f"Pre-validation auto-fixed: {'; '.join(result.auto_fixed)}")

    if not result.is_valid and result.needs_llm_retry:
        feedback_parts = [
            f"PROGRAMMATIC VALIDATION FAILED:\n"
        ]
        for e in result.errors:
            if e.severity == "error":
                feedback_parts.append(f"  ✗ {e.issue}")
        update["editor_retry_feedback"] = "\n".join(feedback_parts)

    return update


# ═══════════════════════════════════════════════════════════
# Phase B: Post-Parallel Validation
# ═══════════════════════════════════════════════════════════

def post_validate_parallel(state: BroadcastState) -> dict:
    """
    LangGraph node: Validates + auto-fixes Visual + Headline outputs.
    
    Checks:
    ✓ Every segment has a visual assignment
    ✓ Every segment has a headline assignment
    ✓ No segment has both source_image_url AND ai_support_visual_prompt
    ✓ Last segment uses "fade_out" transition
    ✓ All headlines ≤ 40 chars, subheadlines ≤ 60 chars
    ✓ No adjacent duplicate headlines
    ✓ Source images used ≤ 2 times each
    ✓ No more than 2 consecutive segments with same layout
    
    Auto-fixes:
    - Truncates headlines/subheadlines that exceed limits
    - Forces last segment to fade_out if missing
    - Removes duplicate source_image_url/ai_support_visual_prompt pairs
    """
    result = ValidationResult()
    segments = state.get("segments", [])
    visual_plan = list(state.get("visual_plan", []))
    headline_plan = list(state.get("headline_plan", []))
    errors_log = []

    segment_ids = {s["segment_id"] for s in segments}

    # ══════════════════════════════════
    # Visual Validation
    # ══════════════════════════════════

    if not visual_plan:
        result.errors.append(ValidationError(
            "visual_plan", "No visual plan generated", "error", False
        ))
        result.needs_llm_retry.append("visual")
    else:
        # Check: every segment has a visual
        planned_visual_ids = {v["segment_id"] for v in visual_plan}
        missing_visual = segment_ids - planned_visual_ids
        if missing_visual:
            result.errors.append(ValidationError(
                "visual_plan", f"Missing visual for segments: {missing_visual}", "error", False
            ))
            result.needs_llm_retry.append("visual")

        # Auto-fix: mutual exclusivity of source_image_url / ai_support_visual_prompt
        for v in visual_plan:
            if v.get("source_image_url") and v.get("ai_support_visual_prompt"):
                v["ai_support_visual_prompt"] = None  # Prefer source image
                result.auto_fixed.append(
                    f"Segment {v['segment_id']}: removed AI prompt (source image takes priority)"
                )

        # Auto-fix: last segment must be fade_out
        if visual_plan and visual_plan[-1].get("transition") != "fade_out":
            old_transition = visual_plan[-1].get("transition", "cut")
            visual_plan[-1]["transition"] = "fade_out"
            result.auto_fixed.append(
                f"Last segment transition changed from '{old_transition}' to 'fade_out'"
            )

        # Check: source images used ≤ 2 times each
        source_usage = {}
        for v in visual_plan:
            url = v.get("source_image_url")
            if url:
                source_usage[url] = source_usage.get(url, 0) + 1
        overused = {url: cnt for url, cnt in source_usage.items() if cnt > 2}
        if overused:
            result.errors.append(ValidationError(
                "visual_plan",
                f"Source images used more than 2 times: {len(overused)} images",
                "warning", False
            ))

        # Check: no more than 2 consecutive same layouts
        for i in range(2, len(visual_plan)):
            if (visual_plan[i].get("layout") == visual_plan[i-1].get("layout") ==
                    visual_plan[i-2].get("layout")):
                result.errors.append(ValidationError(
                    "visual_plan",
                    f"Segments {visual_plan[i-2]['segment_id']}-{visual_plan[i]['segment_id']} "
                    f"all use layout '{visual_plan[i]['layout']}'",
                    "warning", False
                ))
                break

    # ══════════════════════════════════
    # Headline Validation
    # ══════════════════════════════════

    if not headline_plan:
        result.errors.append(ValidationError(
            "headline_plan", "No headline plan generated", "error", False
        ))
        result.needs_llm_retry.append("headline")
    else:
        # Check: every segment has headlines
        planned_headline_ids = {h["segment_id"] for h in headline_plan}
        missing_headline = segment_ids - planned_headline_ids
        if missing_headline:
            result.errors.append(ValidationError(
                "headline_plan", f"Missing headlines for segments: {missing_headline}", "error", False
            ))
            result.needs_llm_retry.append("headline")

        # Auto-fix: truncate overlength headlines
        for h in headline_plan:
            headline = h.get("main_headline", "")
            if len(headline) > 40:
                h["main_headline"] = headline[:37] + "..."
                result.auto_fixed.append(
                    f"Segment {h['segment_id']}: headline truncated from {len(headline)} to 40 chars"
                )
            subheadline = h.get("subheadline", "")
            if len(subheadline) > 60:
                h["subheadline"] = subheadline[:57] + "..."
                result.auto_fixed.append(
                    f"Segment {h['segment_id']}: subheadline truncated from {len(subheadline)} to 60 chars"
                )

        # Check: no adjacent duplicate headlines
        for i in range(1, len(headline_plan)):
            if (headline_plan[i].get("main_headline") ==
                    headline_plan[i-1].get("main_headline")):
                result.errors.append(ValidationError(
                    "headline_plan",
                    f"Segments {headline_plan[i-1]['segment_id']} & {headline_plan[i]['segment_id']} "
                    f"share headline: '{headline_plan[i]['main_headline']}'",
                    "error", False
                ))
                if "headline" not in result.needs_llm_retry:
                    result.needs_llm_retry.append("headline")

    # ── Build state update ──
    update = {
        "visual_plan": visual_plan,
        "headline_plan": headline_plan,
        "post_validation_errors": [e.to_dict() for e in result.errors],
        "errors": errors_log,
    }

    if result.auto_fixed:
        errors_log.append(f"Post-validation auto-fixed: {'; '.join(result.auto_fixed)}")

    # Build targeted retry feedback for failing agents
    if "visual" in result.needs_llm_retry:
        visual_issues = [e.issue for e in result.errors
                        if "visual" in e.field and e.severity == "error"]
        if visual_issues:
            update["visual_retry_feedback"] = (
                "PROGRAMMATIC VALIDATION FAILED:\n" +
                "\n".join(f"  ✗ {issue}" for issue in visual_issues)
            )

    if "headline" in result.needs_llm_retry:
        headline_issues = [e.issue for e in result.errors
                          if "headline" in e.field and e.severity == "error"]
        if headline_issues:
            update["headline_retry_feedback"] = (
                "PROGRAMMATIC VALIDATION FAILED:\n" +
                "\n".join(f"  ✗ {issue}" for issue in headline_issues)
            )

    return update


# ═══════════════════════════════════════════════════════════
# Route Helpers
# ═══════════════════════════════════════════════════════════

def route_after_pre_validation(state: BroadcastState) -> str:
    """Routes after pre-validation: clean → fan-out, errors → retry editor."""
    pre_errors = state.get("pre_validation_errors", [])
    has_critical = any(e.get("severity") == "error" for e in pre_errors)

    if has_critical:
        editor_retries = state.get("editor_retry_count", 0)
        if editor_retries >= 5:
            return "continue"  # Budget exhausted, best effort
        return "retry_editor"

    return "continue"


def route_after_post_validation(state: BroadcastState) -> str:
    """Routes after post-validation: clean → QA, errors → retry targets."""
    post_errors = state.get("post_validation_errors", [])
    has_critical = any(e.get("severity") == "error" for e in post_errors)

    if not has_critical:
        return "qa"

    # Determine which agents need retry
    visual_fail = any("visual" in e.get("field", "") and e.get("severity") == "error"
                      for e in post_errors)
    headline_fail = any("headline" in e.get("field", "") and e.get("severity") == "error"
                        for e in post_errors)

    visual_budget = state.get("visual_retry_count", 0) < 5
    headline_budget = state.get("headline_retry_count", 0) < 5

    if visual_fail and headline_fail and visual_budget and headline_budget:
        return "retry_parallel"
    elif visual_fail and visual_budget:
        return "retry_visual"
    elif headline_fail and headline_budget:
        return "retry_headline"

    return "qa"  # Budgets exhausted, let QA decide


# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════

def _seconds_to_mmss(seconds: int) -> str:
    """Convert seconds to MM:SS format."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"
