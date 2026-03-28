"""
Agent 1: Article Extraction Agent

Fetches a news article URL, extracts clean text, title, metadata, and images.
This agent uses pure tooling (no LLM) — deterministic extraction only.
"""

from src.state import BroadcastState
from src.tools.scraper import fetch_article
from src.tools.image_extractor import extract_images


def extraction_agent(state: BroadcastState) -> dict:
    """
    LangGraph node: Extracts article content from the URL.
    
    Reads: article_url
    Writes: source_title, article_text, source_images, extraction_metadata
    """
    url = state["article_url"]
    errors = []

    # ── Step 1: Fetch and extract article ──
    article = fetch_article(url)

    if not article["success"]:
        error_msg = article.get("error", "Unknown extraction error")
        errors.append(f"Extraction warning: {error_msg}")

        # If we have partial text, still proceed with it
        if not article["text"]:
            errors.append("CRITICAL: No article text could be extracted. Pipeline may produce poor results.")

    # ── Step 2: Extract images from HTML ──
    source_images = []
    if article["html"]:
        try:
            source_images = extract_images(article["html"], url)
        except Exception as e:
            errors.append(f"Image extraction error: {str(e)}")

    # ── Step 3: Validate extraction quality ──
    text = article["text"] or ""
    word_count = len(text.split())

    if word_count < 100:
        errors.append(
            f"Article text is only {word_count} words. "
            f"Minimum recommended is 100 words for a quality broadcast script."
        )

    title = article["title"] or "Untitled Article"
    if title == "Untitled Article":
        errors.append("Could not extract article title. Using fallback.")

    # ── Return state updates ──
    return {
        "source_title": title,
        "article_text": text,
        "source_images": source_images,
        "extraction_metadata": article.get("metadata", {
            "author": "Unknown",
            "date": "Unknown",
            "source_name": "Unknown",
        }),
        "errors": errors,
    }
