"""
Article scraper tool — fetches and extracts clean article text from a URL.

Uses trafilatura for high-accuracy text extraction with requests+BeautifulSoup
as fallback. No LLM calls — pure deterministic tooling.
"""

import requests
import trafilatura
from typing import Optional


def fetch_article(url: str) -> dict:
    """
    Fetches a news article URL and extracts clean text, title, and metadata.

    Args:
        url: The public news article URL.

    Returns:
        dict with keys:
            - title: str — article headline
            - text: str — cleaned article body
            - html: str — raw HTML (for image extraction)
            - metadata: dict — {author, date, source_name}
            - success: bool
            - error: str | None
    """
    result = {
        "title": "",
        "text": "",
        "html": "",
        "metadata": {},
        "success": False,
        "error": None,
    }

    try:
        # ── Step 1: Fetch raw HTML ──
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            # Fallback: use requests with User-Agent
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            downloaded = response.text

        result["html"] = downloaded

        # ── Step 2: Extract article text ──
        extracted_text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )

        if not extracted_text or len(extracted_text.split()) < 50:
            result["error"] = (
                f"Extraction returned insufficient text "
                f"({len(extracted_text.split()) if extracted_text else 0} words). "
                f"Article may be paywalled or dynamic."
            )
            # Still try to use what we got
            if extracted_text:
                result["text"] = extracted_text

        else:
            result["text"] = extracted_text
            result["success"] = True

        # ── Step 3: Extract metadata ──
        metadata = trafilatura.extract_metadata(downloaded)
        if metadata:
            result["title"] = metadata.title or ""
            result["metadata"] = {
                "author": metadata.author or "Unknown",
                "date": str(metadata.date) if metadata.date else "Unknown",
                "source_name": metadata.sitename or _extract_domain(url),
            }
        else:
            result["title"] = _extract_title_fallback(downloaded)
            result["metadata"] = {
                "author": "Unknown",
                "date": "Unknown",
                "source_name": _extract_domain(url),
            }

    except requests.exceptions.RequestException as e:
        result["error"] = f"HTTP error fetching URL: {str(e)}"
    except Exception as e:
        result["error"] = f"Extraction error: {str(e)}"

    return result


def _extract_domain(url: str) -> str:
    """Extract domain name from URL as fallback source name."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain
    except Exception:
        return "Unknown Source"


def _extract_title_fallback(html: str) -> str:
    """Extract title from <title> tag as fallback."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
    except Exception:
        pass
    return "Untitled Article"
