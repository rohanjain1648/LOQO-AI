"""
Image extraction tool — pulls article images from raw HTML.

Filters out noise (icons, ads, tracking pixels) and captures surrounding
context text for each image to enable smart matching in the Visual Agent.
"""

from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List


# Patterns that indicate non-article images (ads, icons, trackers)
NOISE_PATTERNS = [
    "logo", "icon", "avatar", "sprite", "pixel", "tracking",
    "ad-", "advert", "banner", "badge", "button", "widget",
    "social", "share", "facebook", "twitter", "instagram",
    "loading", "spinner", "placeholder", "blank", "spacer",
    "emoji", "favicon",
]

# Minimum dimensions to filter out tiny images
MIN_WIDTH = 200
MIN_HEIGHT = 150


def extract_images(html: str, base_url: str) -> List[dict]:
    """
    Extract article-relevant images from HTML.

    Args:
        html: Raw HTML string of the article page.
        base_url: The article URL (used to resolve relative image paths).

    Returns:
        List of dicts, each with:
            - url: str — absolute image URL
            - alt: str — alt text
            - context: str — surrounding paragraph text
    """
    soup = BeautifulSoup(html, "html.parser")
    images = []
    seen_urls = set()

    for img_tag in soup.find_all("img"):
        src = img_tag.get("src", "")
        if not src:
            continue

        # ── Resolve to absolute URL ──
        abs_url = _resolve_url(src, base_url)
        if not abs_url or abs_url in seen_urls:
            continue

        # ── Filter noise ──
        if _is_noise_image(abs_url, img_tag):
            continue

        # ── Filter by dimensions ──
        if _is_too_small(img_tag):
            continue

        # ── Filter data URIs and SVGs ──
        if abs_url.startswith("data:") or abs_url.endswith(".svg"):
            continue

        seen_urls.add(abs_url)

        # ── Extract context ──
        alt_text = img_tag.get("alt", "").strip()
        context = _extract_context(img_tag)

        images.append({
            "url": abs_url,
            "alt": alt_text or "No description",
            "context": context,
        })

    return images


def _resolve_url(src: str, base_url: str) -> str:
    """Resolve a potentially relative URL to an absolute one."""
    try:
        if src.startswith(("http://", "https://")):
            return src
        return urljoin(base_url, src)
    except Exception:
        return ""


def _is_noise_image(url: str, img_tag) -> bool:
    """Check if image URL or attributes match noise patterns."""
    url_lower = url.lower()
    
    # Check URL path
    for pattern in NOISE_PATTERNS:
        if pattern in url_lower:
            return True

    # Check class and id attributes
    for attr in ["class", "id"]:
        attr_val = img_tag.get(attr, "")
        if isinstance(attr_val, list):
            attr_val = " ".join(attr_val)
        attr_lower = attr_val.lower()
        for pattern in NOISE_PATTERNS:
            if pattern in attr_lower:
                return True

    return False


def _is_too_small(img_tag) -> bool:
    """Check if image has explicit small dimensions."""
    width = img_tag.get("width", "")
    height = img_tag.get("height", "")

    try:
        if width and int(str(width).replace("px", "")) < MIN_WIDTH:
            return True
    except (ValueError, TypeError):
        pass

    try:
        if height and int(str(height).replace("px", "")) < MIN_HEIGHT:
            return True
    except (ValueError, TypeError):
        pass

    return False


def _extract_context(img_tag) -> str:
    """Extract surrounding text context for the image."""
    context_parts = []

    # Check for figcaption (most semantic)
    figure = img_tag.find_parent("figure")
    if figure:
        figcaption = figure.find("figcaption")
        if figcaption:
            context_parts.append(figcaption.get_text(strip=True))

    # Check surrounding paragraphs
    parent = img_tag.parent
    if parent:
        # Previous sibling paragraph
        prev = parent.find_previous_sibling("p")
        if prev:
            text = prev.get_text(strip=True)
            if text and len(text) > 20:
                context_parts.append(text[:200])

        # Next sibling paragraph
        next_sib = parent.find_next_sibling("p")
        if next_sib:
            text = next_sib.get_text(strip=True)
            if text and len(text) > 20:
                context_parts.append(text[:200])

    return " | ".join(context_parts) if context_parts else "No context available"
