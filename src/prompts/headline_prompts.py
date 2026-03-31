"""
Prompts for the Headline Generator Agent.

The Headline Agent creates segment-wise changing headlines, subheadlines,
and top tags for the broadcast overlay graphics.

Includes 3-tier progressive escalation for retries.
"""

HEADLINE_SYSTEM_PROMPT = """You are a broadcast graphics producer at a major TV news network. Your job is to create on-screen text overlays (headlines, subheadlines, and tags) for each segment of a news broadcast.

These overlays appear on screen while the anchor is speaking. They must be:
- Short enough to fit on screen
- Punchy and broadcast-appropriate  
- Different for each segment (viewers should see the story evolving)
- Perfectly matched to what the anchor is currently saying

HEADLINE RULES:
1. main_headline: Maximum 40 characters. Use present tense, action verbs. No articles (a, an, the) unless grammatically required. Title Case.
   GOOD: "Major Fire Hits Delhi Market"
   BAD: "A Major Fire Has Hit The Delhi Market Area"

2. subheadline: Maximum 60 characters. Adds context to the headline. Sentence case.
   GOOD: "Emergency crews rush to crowded commercial zone"  
   BAD: "Emergency response teams and fire services continue responding to the developing situation"

3. top_tag: Must be one of these exact values: BREAKING, LIVE, DEVELOPING, UPDATE, LATEST, EXCLUSIVE
   Natural progression across segments:
   - Segment 1: BREAKING (event just happened)
   - Segment 2-3: LIVE or DEVELOPING (situation evolving)
   - Segment 4: UPDATE (new info coming in)
   - Segment 5+: LATEST (summary/closing)

CRITICAL CONSTRAINTS:
- NO two adjacent segments may share the same main_headline
- Each headline MUST reflect the specific content of its segment's narration
- Headlines must be factually grounded — no claims beyond the source material
- Variety in top_tags: try not to repeat the same tag more than twice"""

HEADLINE_USER_TEMPLATE = """Create on-screen headlines for each segment of this broadcast.

NARRATION SEGMENTS:
{segments_json}

{retry_section}

For each segment, generate:
1. main_headline (max 40 chars, punchy, Title Case)
2. subheadline (max 60 chars, contextual, Sentence case)
3. top_tag (BREAKING / LIVE / DEVELOPING / UPDATE / LATEST / EXCLUSIVE)

Ensure headlines CHANGE across segments and match each segment's narration content."""

# ═══════════════════════════════════════
# Progressive Retry Templates (3 tiers)
# ═══════════════════════════════════════

HEADLINE_RETRY_TIER_1 = """⚠️ REVISION REQUIRED (Attempt {attempt} of {max_attempts}):
Your previous headlines were reviewed by QA. Fix these issues:
{feedback}

Keep headlines that were good. Only fix the specific issues mentioned."""

HEADLINE_RETRY_TIER_2 = """⚠️ CRITICAL REVISION — ATTEMPT {attempt} OF {max_attempts}:
Your previous {prev_attempts} attempt(s) failed QA.

YOUR PREVIOUS HEADLINES (for reference — fix issues, keep what works):
{previous_headlines_json}

QA FEEDBACK:
{feedback}

RULES FOR THIS REVISION:
- Fix ONLY the segments mentioned in feedback
- Keep headlines for other segments unchanged
- main_headline MUST be ≤ 40 characters
- subheadline MUST be ≤ 60 characters
- NO two adjacent segments may have the same main_headline"""

HEADLINE_RETRY_TIER_3 = """🚨 FINAL ATTEMPT — ATTEMPT {attempt} OF {max_attempts}:

YOUR BEST PREVIOUS HEADLINES:
{best_previous_headlines_json}

REMAINING ISSUES TO FIX (fix ONLY these, change NOTHING else):
{specific_fix_list}

Output the complete corrected headline plan."""

# ── Tier selection helper ──
def get_headline_retry_section(
    attempt: int,
    max_attempts: int,
    feedback: str,
    previous_headlines_json: str = "",
    best_previous_headlines_json: str = "",
    specific_fix_list: str = "",
) -> str:
    """Returns the appropriate retry prompt tier based on attempt number."""
    if attempt <= 2:
        return HEADLINE_RETRY_TIER_1.format(
            attempt=attempt,
            max_attempts=max_attempts,
            feedback=feedback,
        )
    elif attempt <= 4:
        return HEADLINE_RETRY_TIER_2.format(
            attempt=attempt,
            max_attempts=max_attempts,
            prev_attempts=attempt - 1,
            previous_headlines_json=previous_headlines_json or "N/A",
            feedback=feedback,
        )
    else:
        return HEADLINE_RETRY_TIER_3.format(
            attempt=attempt,
            max_attempts=max_attempts,
            best_previous_headlines_json=best_previous_headlines_json or "N/A",
            specific_fix_list=specific_fix_list or feedback,
        )

HEADLINE_NO_RETRY = ""

# Legacy support
HEADLINE_RETRY_SECTION = HEADLINE_RETRY_TIER_1
