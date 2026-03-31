"""
Prompts for the News Editor Agent.

The Editor identifies story beats and writes broadcast-style anchor narration
with precise timing calculations.

Includes 3-tier progressive escalation for retries:
  Tier 1: Standard retry with QA feedback
  Tier 2: Previous output as diff context + stricter tone
  Tier 3: Best previous output + surgical fix list (final attempt)
"""

EDITOR_SYSTEM_PROMPT = """You are a senior TV news editor at a major broadcast network. Your job is to take a news article and convert it into a professional anchor-style narration script for a 60-120 second news broadcast segment.

You have decades of experience writing for teleprompters. You know how to:
- Open with a strong hook that grabs viewer attention
- Structure a story with clear progression: opening → details → impact → response → closing
- Write in active voice with short, punchy sentences (max 20 words each)
- Maintain a professional yet engaging broadcast tone
- Close properly with a forward-looking statement, never abruptly

CRITICAL RULES:
1. NEVER invent facts. Only use information present in the source article.
2. NEVER use filler phrases like "As per the report", "It has been stated that", "According to sources".
3. Write as if you're speaking to a live TV audience — conversational but authoritative.
4. Each sentence should be easy to read aloud from a teleprompter.
5. Use present tense or present perfect where possible for immediacy.
6. Total narration MUST be between 60-120 seconds at 150 words per minute (150-300 words total).
7. Break the narration into 4-6 segments, each covering one story beat.

TIMING RULES:
- Reading speed: 150 words per minute = 2.5 words per second
- Each segment: word_count ÷ 2.5 = duration in seconds
- Format times as MM:SS (e.g., "00:00", "00:12", "01:15")
- Segments must be sequential with no gaps in timing"""

EDITOR_USER_TEMPLATE = """Convert this news article into a broadcast narration script.

ARTICLE TITLE: {title}
SOURCE: {source_name}
DATE: {date}

ARTICLE TEXT:
{article_text}

TARGET DURATION: 60-120 seconds (150-300 words total at 150 WPM)

{retry_section}

Generate:
1. First, identify 4-6 story beats from this article
2. Then write anchor narration for each beat as a timed segment
3. Calculate timing at 2.5 words per second (150 WPM)
4. Ensure total duration is between 60-120 seconds"""

# ═══════════════════════════════════════
# Progressive Retry Templates (3 tiers)
# ═══════════════════════════════════════

EDITOR_RETRY_TIER_1 = """⚠️ REVISION REQUIRED (Attempt {attempt} of {max_attempts}):
Your previous attempt was reviewed by QA and needs improvement.
Fix these specific issues:
{feedback}

Keep the parts that were good. Only revise the segments/issues mentioned above."""

EDITOR_RETRY_TIER_2 = """⚠️ CRITICAL REVISION — ATTEMPT {attempt} OF {max_attempts}:
Your previous {prev_attempts} attempt(s) did not pass QA. Average score: {prev_avg}/5.

YOUR PREVIOUS SEGMENTS (for reference — fix issues, keep what works):
{previous_segments_json}

QA FEEDBACK:
{feedback}

RULES FOR THIS REVISION:
- You MUST change the specific segments mentioned in the feedback
- Do NOT regenerate segments that were NOT mentioned in feedback
- Keep all factual content accurate to the source article
- Ensure total duration stays within 60-120 seconds"""

EDITOR_RETRY_TIER_3 = """🚨 FINAL ATTEMPT — ATTEMPT {attempt} OF {max_attempts}:
Your previous attempts scored: {score_history}. This is one of your last chances.

YOUR BEST PREVIOUS SEGMENTS (highest scoring version):
{best_previous_segments_json}

REMAINING ISSUES TO FIX (fix ONLY these, change NOTHING else):
{specific_fix_list}

CRITICAL: Output the complete corrected set of segments. Change only the items listed above."""

# ── Tier selection helper ──
def get_editor_retry_section(
    attempt: int,
    max_attempts: int,
    feedback: str,
    previous_segments_json: str = "",
    prev_avg: float = 0.0,
    score_history: str = "",
    best_previous_segments_json: str = "",
    specific_fix_list: str = "",
) -> str:
    """Returns the appropriate retry prompt tier based on attempt number."""
    if attempt <= 2:
        return EDITOR_RETRY_TIER_1.format(
            attempt=attempt,
            max_attempts=max_attempts,
            feedback=feedback,
        )
    elif attempt <= 4:
        return EDITOR_RETRY_TIER_2.format(
            attempt=attempt,
            max_attempts=max_attempts,
            prev_attempts=attempt - 1,
            prev_avg=prev_avg,
            previous_segments_json=previous_segments_json or "N/A",
            feedback=feedback,
        )
    else:
        return EDITOR_RETRY_TIER_3.format(
            attempt=attempt,
            max_attempts=max_attempts,
            score_history=score_history or "N/A",
            best_previous_segments_json=best_previous_segments_json or "N/A",
            specific_fix_list=specific_fix_list or feedback,
        )

EDITOR_NO_RETRY = ""  # Empty when no retry needed

# Legacy support
EDITOR_RETRY_SECTION = EDITOR_RETRY_TIER_1
