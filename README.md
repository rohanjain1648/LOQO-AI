# 📺 LOQO AI — News URL to Dynamic Broadcast Screenplay Generator

A multi-agent AI pipeline that converts a public news article URL into a **60–120 second TV news broadcast screenplay** — complete with timed narration, segment-wise changing headlines, visual plans, and QA scoring.

## ✨ Features

- **4 AI Agents** orchestrated with LangGraph
- **Parallel execution** — Visual + Headline agents run simultaneously
- **Conditional routing** — QA agent triggers targeted retries
- **Structured output** — JSON + human-readable screenplay
- **LangFuse observability** — full trace visibility
- **Streamlit web UI** — real-time progress + rich display
- **Google Gemini 2.0 Flash** — fast, structured LLM output

## 🏗️ Architecture

```
START → Article Extraction → News Editor
    → [Visual Packaging ∥ Headline Generation]  (parallel)
    → QA Review
    → CONDITIONAL: finalize | retry_editor | retry_visual | retry_headline
    → Format Output → END
```

### Agents

| # | Agent | Purpose |
|---|---|---|
| 1 | **Article Extraction** | Fetches URL, extracts text + images (no LLM) |
| 2 | **News Editor** | Writes broadcast narration in timed segments |
| 3a | **Visual Packaging** | Plans layouts, image assignments, transitions |
| 3b | **Headline Generator** | Creates per-segment headlines + tags |
| 4 | **QA Evaluation** | Scores 5 criteria, routes retries |

### QA Scoring (1-5 per criterion)

- Story Structure & Flow
- Hook & Engagement
- Narration Quality
- Visual Planning & Placement
- Headline / Subheadline Quality

**Pass rule:** No criterion below 3, overall average ≥ 4.0, all boolean checks pass.

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd "loqo ai"
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your keys:
#   GOOGLE_API_KEY=your-gemini-api-key
#   LANGFUSE_PUBLIC_KEY=pk-lf-...
#   LANGFUSE_SECRET_KEY=sk-lf-...
```

### 3. Run (CLI)

```bash
python main.py https://example.com/news/article
```

### 4. Run (Streamlit Web UI)

```bash
streamlit run app.py
```

## 📂 Project Structure

```
loqo ai/
├── app.py                    # Streamlit web interface
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
│
├── src/
│   ├── config.py             # Gemini + LangFuse setup
│   ├── state.py              # Central state schema
│   ├── graph.py              # LangGraph assembly
│   │
│   ├── agents/
│   │   ├── extraction.py     # Agent 1: Article extraction
│   │   ├── editor.py         # Agent 2: News Editor
│   │   ├── visual.py         # Agent 3a: Visual Packaging
│   │   ├── headline.py       # Agent 3b: Headline Generator
│   │   └── qa.py             # Agent 4: QA Evaluation
│   │
│   ├── tools/
│   │   ├── scraper.py        # URL fetch + text extraction
│   │   └── image_extractor.py # Image extraction + filtering
│   │
│   ├── prompts/
│   │   ├── editor_prompts.py
│   │   ├── visual_prompts.py
│   │   ├── headline_prompts.py
│   │   └── qa_prompts.py
│   │
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   │
│   └── output/
│       ├── formatter.py      # JSON + screenplay formatting
│       └── templates.py      # Screenplay text templates
```

## 📊 Output Format

### JSON
```json
{
  "article_url": "https://...",
  "source_title": "...",
  "video_duration_sec": 75,
  "segments": [
    {
      "segment_id": 1,
      "start_time": "00:00",
      "end_time": "00:12",
      "layout": "anchor_left + source_visual_right",
      "anchor_narration": "Good evening...",
      "main_headline": "Major Fire Hits Market",
      "subheadline": "Emergency crews respond",
      "top_tag": "BREAKING",
      "left_panel": "AI anchor in studio",
      "right_panel": "Source image of fire",
      "source_image_url": "https://...",
      "ai_support_visual_prompt": null,
      "transition": "cut"
    }
  ]
}
```

## 🔄 Conditional Edge Routing

```
QA passes → finalize (no retries)
QA fails narration → retry only Editor
QA fails visuals → retry only Visual agent
QA fails headlines → retry only Headline agent
Max 3 retries → force finalize with best effort
```

## 📡 LangFuse Observability

All pipeline runs are traced in LangFuse with:
- Per-agent execution spans
- Token usage and latency
- QA scores pushed as custom metrics
- Retry metadata

## Tech Stack

- **LLM:** Google Gemini 2.0 Flash (via langchain-google-genai)
- **Orchestration:** LangGraph
- **Observability:** LangFuse
- **Extraction:** Trafilatura + BeautifulSoup4
- **Models:** Pydantic v2
- **Frontend:** Streamlit
