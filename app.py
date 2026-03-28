"""
LOQO AI — News URL to Broadcast Screenplay Generator
Streamlit Web Interface

Run: streamlit run app.py
"""

import json
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──
st.set_page_config(
    page_title="LOQO AI — Broadcast Screenplay Generator",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
    
    .agent-card.done {
        border-left-color: #10b981;
    }
    
    .agent-card.running {
        border-left-color: #f59e0b;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .agent-card.waiting {
        border-left-color: #4b5563;
        opacity: 0.6;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        text-align: center;
        min-width: 140px;
    }
    
    .segment-card {
        background: #0f172a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid #1e293b;
    }
    
    .tag-badge {
        display: inline-block;
        background: #dc2626;
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    
    .headline-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.3rem 0;
    }
    
    .subheadline-text {
        font-size: 0.95rem;
        color: #94a3b8;
    }
    
    .narration-block {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.8rem;
        font-style: italic;
        color: #e2e8f0;
        line-height: 1.6;
        border-left: 3px solid #667eea;
    }
    
    .visual-info {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Header ──
    st.markdown('<div class="main-title">📺 LOQO AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">News URL to Dynamic Broadcast Screenplay Generator</div>', unsafe_allow_html=True)

    # ── URL Input ──
    col1, col2 = st.columns([5, 1])
    with col1:
        url = st.text_input(
            "🔗 News Article URL",
            placeholder="https://example.com/news/your-article",
            label_visibility="collapsed",
        )
    with col2:
        generate_btn = st.button("▶ Generate", type="primary", use_container_width=True)

    if generate_btn and url:
        _run_pipeline(url)
    elif generate_btn and not url:
        st.error("Please enter a news article URL.")

    # ── Display results if available ──
    if "result" in st.session_state:
        _display_results(st.session_state["result"])


def _run_pipeline(url: str):
    """Execute the pipeline with real-time progress tracking."""
    from src.graph import app
    from src.config import langfuse_handler

    # ── Agent progress tracking ──
    agents = [
        ("extract_article", "📥 Article Extraction"),
        ("generate_narration", "✍️ Narration Generation"),
        ("plan_visuals", "🎬 Visual Packaging"),
        ("generate_headlines", "📰 Headline Generation"),
        ("review_quality", "🔍 QA Review"),
        ("format_output", "📄 Final Output"),
    ]
    agent_status = {name: "waiting" for name, _ in agents}

    # ── Progress display ──
    st.markdown("---")
    progress_container = st.container()

    with progress_container:
        st.subheader("⚡ Pipeline Progress")
        progress_bar = st.progress(0)
        status_placeholders = {}

        for name, label in agents:
            status_placeholders[name] = st.empty()
            status_placeholders[name].markdown(
                f'<div class="agent-card waiting">⬜ {label} — Waiting</div>',
                unsafe_allow_html=True,
            )

    retry_info = st.empty()

    # ── Stream execution ──
    completed_nodes = set()
    result = None

    try:
        for event in app.stream(
            {"article_url": url, "retry_count": 0, "errors": []},
            config={"callbacks": [langfuse_handler]},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue

                completed_nodes.add(node_name)

                # Update completed node
                for name, label in agents:
                    if name == node_name:
                        status_placeholders[name].markdown(
                            f'<div class="agent-card done">✅ {label} — Done</div>',
                            unsafe_allow_html=True,
                        )

                # Update progress
                progress = len(completed_nodes) / len(agents)
                progress_bar.progress(min(progress, 1.0))

                # Show retry info if applicable
                retry_count = node_output.get("retry_count", 0) if isinstance(node_output, dict) else 0
                if retry_count > 1:
                    retry_info.info(f"🔄 Retry round {retry_count - 1}/3")

                # Mark next nodes as running
                found_current = False
                for name, label in agents:
                    if name == node_name:
                        found_current = True
                        continue
                    if found_current and name not in completed_nodes:
                        status_placeholders[name].markdown(
                            f'<div class="agent-card running">⏳ {label} — Running</div>',
                            unsafe_allow_html=True,
                        )
                        break

                # Capture result
                result = node_output

        # ── Get final result by invoking once more (stream doesn't return merged state) ──
        # Actually, let's reconstruct from the session
        final_result = app.invoke(
            {"article_url": url, "retry_count": 0, "errors": []},
            config={"callbacks": [langfuse_handler]},
        )

        progress_bar.progress(1.0)
        st.session_state["result"] = final_result
        st.success("✅ Screenplay generated successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Pipeline error: {str(e)}")
        st.exception(e)


def _display_results(result: dict):
    """Display the generated screenplay and JSON."""

    st.markdown("---")

    # ── QA Scores ──
    qa_scores = result.get("qa_scores", {})
    if qa_scores:
        st.subheader("📊 Quality Scores")

        score_cols = st.columns(5)
        score_labels = [
            ("story_structure", "Structure"),
            ("hook_engagement", "Hook"),
            ("narration_quality", "Narration"),
            ("visual_planning", "Visuals"),
            ("headline_quality", "Headlines"),
        ]

        for i, (key, label) in enumerate(score_labels):
            score = qa_scores.get(key, 0)
            stars = "★" * score + "☆" * (5 - score)
            with score_cols[i]:
                st.metric(label, f"{score}/5")
                st.caption(stars)

        # Pass/fail status
        col1, col2 = st.columns(2)
        with col1:
            avg = round(sum(qa_scores.values()) / len(qa_scores), 1) if qa_scores else 0
            status = "✅ PASSED" if result.get("qa_pass") else "⚠️ Best Effort"
            st.metric("Overall Average", f"{avg}/5")
            st.caption(status)
        with col2:
            st.metric("Retries Used", f"{result.get('retry_count', 0)}/3")

    st.markdown("---")

    # ── Output Tabs ──
    tab1, tab2, tab3 = st.tabs(["📄 Screenplay", "📊 JSON Output", "ℹ️ Extraction Info"])

    with tab1:
        _display_screenplay(result)

    with tab2:
        final_json = result.get("final_json", {})
        st.json(final_json)

        # Download button
        json_str = json.dumps(final_json, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download JSON",
            data=json_str,
            file_name="broadcast_screenplay.json",
            mime="application/json",
        )

    with tab3:
        st.markdown(f"**Title:** {result.get('source_title', 'N/A')}")
        st.markdown(f"**URL:** {result.get('article_url', 'N/A')}")

        metadata = result.get("extraction_metadata", {})
        st.markdown(f"**Source:** {metadata.get('source_name', 'N/A')}")
        st.markdown(f"**Author:** {metadata.get('author', 'N/A')}")
        st.markdown(f"**Date:** {metadata.get('date', 'N/A')}")

        images = result.get("source_images", [])
        st.markdown(f"**Images Found:** {len(images)}")
        if images:
            for img in images:
                st.markdown(f"- [{img.get('alt', 'Image')}]({img.get('url', '#')})")

        # Errors/warnings
        errors = result.get("errors", [])
        if errors:
            st.warning(f"⚠️ {len(errors)} warnings during processing:")
            for err in errors:
                st.text(f"  • {err}")


def _display_screenplay(result: dict):
    """Display the screenplay in a rich visual format."""
    final_json = result.get("final_json", {})
    segments = final_json.get("segments", [])

    # ── Header ──
    st.markdown(f"### 📺 {final_json.get('source_title', 'Broadcast Screenplay')}")
    st.markdown(f"**Duration:** {final_json.get('video_duration_sec', 0)} seconds | **Segments:** {len(segments)}")

    # ── Download screenplay ──
    screenplay_text = result.get("final_screenplay", "")
    if screenplay_text:
        st.download_button(
            "📥 Download Screenplay",
            data=screenplay_text,
            file_name="broadcast_screenplay.txt",
            mime="text/plain",
        )

    st.markdown("---")

    # ── Render each segment ──
    for seg in segments:
        tag_colors = {
            "BREAKING": "#dc2626",
            "LIVE": "#059669",
            "DEVELOPING": "#d97706",
            "UPDATE": "#2563eb",
            "LATEST": "#7c3aed",
            "EXCLUSIVE": "#be185d",
        }

        tag = seg.get("top_tag", "LATEST")
        tag_color = tag_colors.get(tag, "#6b7280")

        st.markdown(f"""
<div class="segment-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
        <span style="color:#64748b; font-weight:600;">SEGMENT {seg['segment_id']}</span>
        <span style="color:#64748b;">{seg.get('start_time', '')} – {seg.get('end_time', '')}</span>
    </div>
    <span class="tag-badge" style="background:{tag_color};">{tag}</span>
    <div class="headline-text">{seg.get('main_headline', '')}</div>
    <div class="subheadline-text">{seg.get('subheadline', '')}</div>
    <div class="visual-info">
        🎬 <strong>Layout:</strong> {seg.get('layout', '')} | 
        ⏭️ <strong>Transition:</strong> {seg.get('transition', '')}
    </div>
    <div class="visual-info">
        👤 {seg.get('left_panel', '')} | 
        🖼️ {seg.get('right_panel', '')}
    </div>
    {'<div class="visual-info">📸 <a href="' + seg["source_image_url"] + '" target="_blank">Source Image</a></div>' if seg.get("source_image_url") else ''}
    {'<div class="visual-info">🎨 AI Prompt: <em>' + seg["ai_support_visual_prompt"] + '</em></div>' if seg.get("ai_support_visual_prompt") else ''}
    <div class="narration-block">
        🎙️ "{seg.get('anchor_narration', '')}"
    </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
