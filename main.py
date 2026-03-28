"""
LOQO AI — News URL to Broadcast Screenplay Generator
CLI Entry Point

Usage:
    python main.py <news_article_url>
    python main.py  (will prompt for URL)

Outputs:
    - output.json — Structured JSON screenplay
    - screenplay.txt — Human-readable screenplay
    - Console — Screenplay + QA scores
"""

import sys
import json
from dotenv import load_dotenv


def main():
    load_dotenv()

    # ── Get URL from argument or prompt ──
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("\n🔗 Enter news article URL: ").strip()

    if not url:
        print("❌ No URL provided. Exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📺 LOQO AI — Broadcast Screenplay Generator")
    print(f"{'='*60}")
    print(f"🔗 URL: {url}")
    print(f"{'='*60}\n")

    # ── Import after dotenv to ensure env vars are loaded ──
    from src.graph import app
    from src.config import langfuse_handler

    # ── Run the pipeline ──
    print("🚀 Starting pipeline...\n")

    try:
        result = app.invoke(
            {
                "article_url": url,
                "retry_count": 0,
                "errors": [],
            },
            config={"callbacks": [langfuse_handler]},
        )

        # ── Save JSON output ──
        final_json = result.get("final_json", {})
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        # ── Save screenplay output ──
        final_screenplay = result.get("final_screenplay", "")
        with open("screenplay.txt", "w", encoding="utf-8") as f:
            f.write(final_screenplay)

        # ── Print screenplay to console ──
        print(final_screenplay)

        # ── Summary ──
        print(f"\n{'='*60}")
        print(f"✅ JSON saved to: output.json")
        print(f"✅ Screenplay saved to: screenplay.txt")

        # ── QA Summary ──
        qa_scores = result.get("qa_scores", {})
        if qa_scores:
            avg = round(sum(qa_scores.values()) / len(qa_scores), 1)
            print(f"\n📊 QA Scores: {qa_scores}")
            print(f"📈 Average: {avg}/5")
            print(f"🔄 Retries used: {result.get('retry_count', 0)}/3")
            print(f"{'✅ PASSED' if result.get('qa_pass') else '⚠️ BEST EFFORT (max retries reached)'}")

        # ── Errors ──
        errors = result.get("errors", [])
        if errors:
            print(f"\n⚠️  Warnings ({len(errors)}):")
            for err in errors:
                print(f"   • {err}")

        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
