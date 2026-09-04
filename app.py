import tempfile
import os
import streamlit as st

from src.preprocessing import split_sentences
from src.pipeline import FocusNotesPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FocusNotes AI",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# APPLE-INSPIRED CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>
    /* System Font Stack & Global Resets */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1D1D1F;
        background-color: #F5F5F7;
    }
    
    /* Hide Streamlit default chrome header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container Padding */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    /* Headings */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #1D1D1F !important;
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: #1D1D1F !important;
    }

    .stSubheader {
        font-size: 1.15rem !important;
        color: #86868B !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
    }

    /* Input Card Containers */
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        border-radius: 12px !important;
        border: 1px solid #E5E5EA !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
    }

    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: #0071E3 !important;
        box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.15) !important;
    }

    /* Primary Action Button */
    button[kind="primary"] {
        background-color: #0071E3 !important;
        color: #FFFFFF !important;
        border-radius: 980px !important;
        border: none !important;
        padding: 0.65rem 2rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 12px rgba(0, 113, 227, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    button[kind="primary"]:hover {
        background-color: #0077ED !important;
        transform: scale(1.015) !important;
        box-shadow: 0 6px 16px rgba(0, 113, 227, 0.35) !important;
    }

    /* Sticky Note Apple Cards */
    .stInfo {
        background-color: #FFFFFF !important;
        color: #1D1D1F !important;
        border: 1px solid #E5E5EA !important;
        border-left: 5px solid #0071E3 !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.4rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 0.8rem !important;
        font-size: 0.98rem !important;
        line-height: 1.5 !important;
    }

    /* Subtle Category Pill Badges */
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 980px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .cat-action { background-color: #E8F5E9; color: #2E7D32; }
    .cat-decision { background-color: #E3F2FD; color: #1565C0; }
    .cat-problem { background-color: #FFEBEE; color: #C62828; }
    .cat-idea { background-color: #FFF8E1; color: #F57F17; }
    .cat-info { background-color: #F3E5F5; color: #7B1FA2; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("FocusNotes AI")
st.markdown("<p class='stSubheader'>Context-Aware Conversation & Audio Intelligence</p>", unsafe_allow_html=True)

st.divider()


# ============================================================
# INPUTS
# ============================================================

agenda = st.text_input(
    "Meeting / Conversation Topic",
    placeholder="e.g., Website Performance Optimization"
)

col1, col2 = st.columns(2)

with col1:
    conversation = st.text_area(
        "Conversation Transcript (Text)",
        placeholder="Paste conversation transcript here (or upload audio to auto-transcribe)...",
        height=250
    )

with col2:
    audio_file = st.file_uploader(
        "Optional Audio File (WAV / MP3)",
        type=["wav", "mp3", "m4a"]
    )
    if audio_file is not None:
        st.audio(audio_file)


# ============================================================
# LOAD PIPELINE
# ============================================================

pipeline = FocusNotesPipeline()


# ============================================================
# ANALYZE CONVERSATION
# ============================================================

if st.button(
    "🧠 Analyze Conversation & Audio",
    type="primary"
):

    if not agenda:
        st.warning("Please enter a meeting topic.")
    elif not conversation and audio_file is None:
        st.warning("Please enter a text conversation OR upload an audio file.")
    else:
        temp_audio_path = None
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                tmp.write(audio_file.read())
                temp_audio_path = tmp.name

        with st.spinner("Analyzing conversation & extracting prosodic features..."):
            output = pipeline.analyze(
                agenda=agenda,
                conversation=conversation,
                audio_path=temp_audio_path,
                mode="generative"
            )

        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        results = output["results"]
        sticky_notes = output["sticky_notes"]
        features = output.get("features", {})
        prosody = features.get("prosody", {})

        if output.get("asr_transcript"):
            st.info(f"🎙️ Auto-Transcribed Audio (Whisper ASR):\n{output['asr_transcript']}")

        st.success("Conversation & Prosodic analysis completed successfully!")

        # ====================================================
        # RELEVANCE ANALYSIS & PROSODY
        # ====================================================

        st.divider()
        st.subheader("🧠 Relevance & Prosody Analysis")

        prosody_list = prosody.get("prosodic_features", [])

        for idx, result in enumerate(results):
            sentence = result["sentence"]
            relevance = result["relevance"]
            category = result["category"]

            p_info = ""
            if prosody.get("has_audio") and idx < len(prosody_list):
                p_item = prosody_list[idx]
                p_info = f" | 🎶 *Pitch Var*: `{p_item.get('pitch_variance', 0.0):.2f}` | ⏱️ *Pause*: `{p_item.get('pause_duration', 0.0):.2f}s`"

            if relevance == "RELEVANT":
                st.markdown(f"🟢 **RELEVANT** — `{category}`{p_info}")
                st.write(sentence)
            else:
                st.markdown(f"⚪ **IGNORED**{p_info}")
                st.write(sentence)

        # ====================================================
        # STICKY NOTES
        # ====================================================

        st.divider()
        st.subheader("📝 FocusNotes (Sticky Notes)")

        for cat_name, icon in [("PROBLEM", "🔴 Problems"), ("IDEA", "💡 Ideas"), ("ACTION", "✅ Actions"), ("DECISION", "🎯 Decisions"), ("INFORMATION", "ℹ️ Information")]:
            if cat_name in sticky_notes:
                st.markdown(f"### {icon}")
                for note in sticky_notes[cat_name]:
                    st.info(note)