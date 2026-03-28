"""
Screenplay text templates for human-readable output formatting.
"""

SCREENPLAY_HEADER = """═══════════════════════════════════════════════════════════
📺 BROADCAST SCREENPLAY
═══════════════════════════════════════════════════════════
Source: "{title}"
URL:    {url}
Duration: {duration} seconds | Segments: {segment_count}
═══════════════════════════════════════════════════════════
"""

SEGMENT_TEMPLATE = """
━━━ SEGMENT {segment_id} ━━━ [{start_time} – {end_time}] ━━━━━━━━━━━━━━━━━━━━━━
🏷️  {top_tag}
📰 {main_headline}
   {subheadline}

🎬 Layout: {layout}
👤 Left:  {left_panel}
🖼️  Right: {right_panel}
{image_line}
🎙️ NARRATION:
\"{narration}\"

⏭️  Transition: {transition}
"""

SOURCE_IMAGE_LINE = "📸 Image: {url}\n"

AI_VISUAL_LINE = "🎨 AI Prompt: \"{prompt}\"\n"

NO_IMAGE_LINE = ""

QA_SCORES_TEMPLATE = """
═══════════════════════════════════════════════════════════
📊 QA SCORES
═══════════════════════════════════════════════════════════
Story Structure:  {stars_structure} ({score_structure}/5)
Hook & Engagement: {stars_hook} ({score_hook}/5)
Narration Quality: {stars_narration} ({score_narration}/5)
Visual Planning:   {stars_visual} ({score_visual}/5)
Headline Quality:  {stars_headline} ({score_headline}/5)
Overall Average:   {avg_score}/5 {pass_status}
Retries Used: {retry_count}/3
═══════════════════════════════════════════════════════════
"""


def make_stars(score: int) -> str:
    """Convert a 1-5 score to star representation."""
    filled = "★" * score
    empty = "☆" * (5 - score)
    return filled + empty
