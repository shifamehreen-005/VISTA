# app.py
from __future__ import annotations

import base64
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Video Intelligence with Spatio-Temporal Augmented Retrieval for Egocentric Understanding",
    layout="wide",
    page_icon="▶",  # Favicon (emoji) avoids ERR_FILE_NOT_FOUND for missing favicon.ico
)

# -----------------------------
# Theme: background image + visible text
# -----------------------------
_bgm_b64 = ""
_bgm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgm.jpeg")
if os.path.isfile(_bgm_path):
    with open(_bgm_path, "rb") as _f:
        _bgm_b64 = base64.b64encode(_f.read()).decode("utf-8")

_logo_b64 = ""
_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VISTA_logo.png")
if os.path.isfile(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode("utf-8")

st.markdown(
    """
<style>
/* ---------- Light theme: background image at 50% above cream, below content ---------- */
.stApp {
  position: relative;
  background: #FAF0DC;
  background-attachment: fixed;
}
.stApp::before {
  content: "";
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image: url("data:image/jpeg;base64,__BGM_B64__");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  opacity: 0.3;
  z-index: 0;
  pointer-events: none;
}
.stApp::after {
  content: "";
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(101, 67, 43, 0.05);
  z-index: 0;
  pointer-events: none;
}
[data-testid="stAppViewContainer"],
.block-container,
[data-testid="stHeader"],
section[data-testid="stHeader"] {
  position: relative;
  z-index: 1;
}
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 1.5rem;
  padding-left: 2.5rem;
  padding-right: 2.5rem;
  max-width: 100%;
  box-sizing: border-box;
}

/* ---------- Video player: rounded borders ---------- */
[data-testid="stVideo"],
[data-testid="stVideo"] > div,
[data-testid="stVideo"] video,
.stApp video {
  border-radius: 12px !important;
  overflow: hidden !important;
}
[data-testid="stVideo"] video,
.stApp video {
  display: block !important;
}

/* ---------- Ensure all text is visible (light theme) ---------- */
.stApp p, .stApp span, .stApp label, .stApp div[data-testid="stMarkdown"],
.stApp .stMarkdown, .stApp h1, .stApp h2, .stApp h3 {
  color: #1a1a1e !important;
}
.stApp h1 {
  font-size: 1.75rem !important;
  font-weight: 600 !important;
}
.stApp .stMarkdown strong { color: #0d0d0f !important; }
.stApp a { color: #0066cc !important; }

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Copyright footer */
.app-footer {
  text-align: center;
  font-size: 0.95rem;
  font-weight: 600;
  color: #374151;
  padding: 1.25rem 1rem;
  margin-top: 1.5rem;
  border-top: 1px solid rgba(0, 0, 0, 0.12);
}

/* Hide Streamlit-generated element (st-emotion-cache-fis6aj) */
.st-emotion-cache-fis6aj.ewslnz97,
.st-emotion-cache-fis6aj,
.ewslnz97 {
  display: none !important;
}

/* ---------- Navigation bar: dark translucent (reference style) ---------- */
.stApp header,
header[data-testid="stHeader"],
section[data-testid="stHeader"],
.stAppHeader,
.nav-bar-vibe {
  background: rgba(250, 240, 220, 0.3) !important;
  background-color: rgba(250, 240, 220, 0.3) !important;
  position: relative;
  border: none !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}
.nav-bar-vibe::after {
  content: none;
}
.stApp header::after,
header[data-testid="stHeader"]::after,
.stAppHeader::after {
  content: none;
}
.nav-bar-vibe {
  width: 100vw;
  margin-left: calc(-50vw + 50%);
  margin-top: -1rem;
  margin-bottom: 1.5rem;
  padding: 0.75rem 4rem 1rem 4rem;
  min-height: 64px;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-radius: 0;
  box-sizing: border-box;
}
.nav-bar-vibe .nav-bar-left {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}
.nav-bar-vibe .nav-bar-logo {
  height: 60px;
  width: auto;
  display: block;
  flex-shrink: 0;
  position: relative;
  z-index: 1;
}
.nav-bar-vibe .nav-bar-title {
  color: #1a1a1e !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
  position: relative;
  z-index: 1;
  white-space: nowrap;
}
.nav-bar-vibe .nav-bar-subtitle {
  color: #4a4238 !important;
  font-size: 1rem !important;
  font-weight: 400 !important;
  font-style: italic !important;
  position: relative;
  z-index: 1;
  line-height: 1.3;
  text-align: right;
  max-width: 60%;
}
.stApp header span,
.stApp header svg,
.stAppHeader span,
.stAppHeader svg {
  color: #1a1a1e !important;
}

/* ---------- Chat (left-right, elegant): one box with input inside ---------- */
[data-testid="column"]:has([data-testid="stForm"]),
[data-testid="stHorizontalBlock"] > div:last-child {
  background: #FFFEF8 !important;
  border-radius: 12px !important;
  padding: 1.25rem !important;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
  border: 1px solid rgba(0, 0, 0, 0.04);
  margin-left: 0.25rem;
}
[data-testid="stHorizontalBlock"] > div:first-child {
  padding-left: 0.5rem;
  padding-right: 0.5rem;
  box-sizing: border-box;
}
.chat-panel {
  font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  height: 392px;
  display: flex;
  flex-direction: column;
  background: #FFFEF8;
  border-radius: 12px;
  padding: 1.25rem;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
  border: 1px solid rgba(0, 0, 0, 0.04);
}
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px 0;
  scrollbar-width: thin;
}
.chat-messages::-webkit-scrollbar { width: 5px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.2); border-radius: 3px; }
.chat-row {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 14px;
}
.chat-row.user {
  justify-content: flex-end;
}
.chat-bubble {
  max-width: 78%;
  padding: 10px 14px;
  font-size: 13px;
  line-height: 1.52;
  letter-spacing: 0.01em;
  color: #374151;
  border-radius: 14px;
  word-wrap: break-word;
}
.chat-row .chat-bubble {
  background: #f0f1f5;
  border: 1px solid rgba(0,0,0,0.06);
}
.chat-row.user .chat-bubble {
  background: rgba(99, 102, 241, 0.15);
  color: #1f2937;
  border: 1px solid rgba(99, 102, 241, 0.25);
}
.chat-bubble p { margin: 0; }
.chat-bubble strong { color: #111827; font-weight: 600; }
.chat-bubble code {
  font-size: 12px;
  background: rgba(0,0,0,0.08);
  padding: 2px 5px;
  border-radius: 4px;
  color: #1a1a1e;
}
.chat-meta {
  margin-top: 6px;
  font-size: 11px;
  color: #6b7280;
}

/* ---------- Chat input: no borders, input and button same height in line ---------- */
div[data-testid="stForm"] {
  padding-right: 0 !important;
  margin-top: 0.75rem !important;
  margin-bottom: 0 !important;
}
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
  display: flex !important;
  align-items: center !important;
  gap: 0.5rem !important;
}
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:first-child {
  flex: 1 !important;
  min-width: 0 !important;
}
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:last-child {
  flex: 0 0 64px !important;
  width: 64px !important;
  min-width: 64px !important;
  max-width: 64px !important;
}
div[data-testid="stForm"] .stTextInput > div,
div[data-testid="stForm"] .stTextInput > div > div {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
  height: 44px !important;
  min-height: 44px !important;
  display: flex !important;
  align-items: center !important;
}
div[data-testid="stForm"] input {
  font-size: 13px !important;
  color: #1a1a1e !important;
  background: rgba(255,255,255,0.6) !important;
  background-color: rgba(255,255,255,0.6) !important;
  border: none !important;
  border-radius: 22px !important;
  padding: 0 16px !important;
  height: 44px !important;
  line-height: 44px !important;
  outline: none !important;
  box-shadow: none !important;
  box-sizing: border-box !important;
}
div[data-testid="stForm"] input:-webkit-autofill,
div[data-testid="stForm"] input:-webkit-autofill:hover,
div[data-testid="stForm"] input:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0 9999px rgba(255,255,255,0.6) inset !important;
  box-shadow: 0 0 0 9999px rgba(255,255,255,0.6) inset !important;
  background-color: rgba(255,255,255,0.6) !important;
}
div[data-testid="stForm"] input:focus,
div[data-testid="stForm"] .stTextInput input:focus,
div[data-testid="stForm"] input:invalid,
div[data-testid="stForm"] input:user-invalid,
div[data-testid="stForm"] .stTextInput > div:focus-within,
div[data-testid="stForm"] .stTextInput > div > div:focus-within {
  border: none !important;
  box-shadow: none !important;
}
div[data-testid="stForm"] input::placeholder {
  color: rgba(0, 0, 0, 0.4) !important;
}
/* Send button: same height as input, in line */
section[data-testid="stForm"] button,
div[data-testid="stForm"] button,
[data-testid="stForm"] .stButton > button,
[data-testid="stForm"] button {
  width: 44px !important;
  min-width: 44px !important;
  height: 44px !important;
  padding: 0 !important;
  font-size: 24px !important;
  line-height: 1 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  color: #ffffff !important;
  background: linear-gradient(180deg, #5eb0ff 0%, #0077ed 50%, #0066d6 100%) !important;
  background-color: #0077ed !important;
  border: none !important;
  outline: none !important;
  box-shadow: 0 2px 8px rgba(0, 102, 214, 0.35) !important;
  border-radius: 50% !important;
  flex-shrink: 0 !important;
}
[data-testid="stForm"] button:focus {
  outline: none !important;
  box-shadow: 0 2px 8px rgba(0, 102, 214, 0.35) !important;
}
section[data-testid="stForm"] button:hover,
div[data-testid="stForm"] button:hover,
[data-testid="stForm"] .stButton > button:hover,
[data-testid="stForm"] button:hover {
  color: #ffffff !important;
  background: linear-gradient(180deg, #70bcff 0%, #0088ff 50%, #0077ed 100%) !important;
  background-color: #0088ff !important;
  box-shadow: 0 3px 12px rgba(0, 119, 237, 0.45) !important;
}
/* Remove multiple boxes around send button: strip wrappers */
div[data-testid="stForm"] > div,
div[data-testid="stForm"] [data-testid="stHorizontalBlock"],
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div,
div[data-testid="stForm"] [data-testid="column"]:last-of-type,
[data-testid="stForm"] .stButton,
[data-testid="stForm"] .stButton > div {
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  min-height: 0 !important;
}
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:last-child {
  padding-right: 0 !important;
}
div[data-testid="stForm"] [data-testid="column"]:last-of-type {
  padding-right: 20px !important;
  flex: 0 0 64px !important;
  width: 64px !important;
  min-width: 64px !important;
  max-width: 64px !important;
  box-sizing: border-box !important;
}
div[data-testid="stForm"] [data-testid="column"]:first-of-type {
  flex: 1 !important;
  min-width: 0 !important;
}

/* ---------- File uploader: light theme ---------- */
[data-testid="stFileUploader"] {
  color: #1a1a1e !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] > div {
  background: #ffffff !important;
  border: 1px dashed rgba(0,0,0,0.25) !important;
  border-radius: 12px !important;
  padding: 24px !important;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploader"] > div:hover {
  border-color: rgba(0,0,0,0.4) !important;
  background: #f8f9fc !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] a {
  color: #1a1a1e !important;
  font-size: 14px !important;
  font-weight: 500 !important;
}
[data-testid="stFileUploader"] small {
  color: #6b7280 !important;
  font-size: 13px !important;
}
/* Hide the grey square/icon box in the dropzone */
[data-testid="stFileUploader"] svg {
  display: none !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:first-child,
[data-testid="stFileUploader"] > div > div:first-child:not(:last-child) {
  display: none !important;
}
/* Hide the filename/size box when a file is selected */
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) [data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) > div > section {
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
  min-height: 0 !important;
}
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) [data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) [data-testid="stFileUploaderDropzone"] a,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) [data-testid="stFileUploaderDropzone"] span:not(.stButton *) {
  display: none !important;
}
/* Browse files button only: yellow with black text - target dropzone and first button */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button,
.stFileUploader .stButton:first-of-type > button,
[data-testid="stFileUploader"] .stButton:first-of-type > button {
  background: linear-gradient(180deg, #fde047 0%, #eab308 50%, #ca8a04 100%) !important;
  background-color: #eab308 !important;
  color: #000000 !important;
  border: none !important;
  font-weight: 600 !important;
  padding: 10px 20px !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 6px rgba(202, 138, 4, 0.35) !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button *,
.stFileUploader .stButton:first-of-type > button *,
[data-testid="stFileUploader"] .stButton:first-of-type > button * {
  color: #000000 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover,
.stFileUploader .stButton:first-of-type > button:hover,
[data-testid="stFileUploader"] .stButton:first-of-type > button:hover {
  background: linear-gradient(180deg, #fef08a 0%, #facc15 50%, #eab308 100%) !important;
  background-color: #facc15 !important;
  color: #000000 !important;
  box-shadow: 0 3px 10px rgba(234, 179, 8, 0.45) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover *,
.stFileUploader .stButton:first-of-type > button:hover *,
[data-testid="stFileUploader"] .stButton:first-of-type > button:hover * {
  color: #000000 !important;
}
/* Remove file button: dustbin icon - 2nd button when file is added */
.stFileUploader:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button {
  min-width: 0 !important;
  width: 36px !important;
  padding: 6px 8px !important;
  background: transparent !important;
  background-color: transparent !important;
  color: #6b7280 !important;
  font-size: 18px !important;
  font-weight: 400 !important;
  line-height: 1 !important;
  box-shadow: none !important;
  border-radius: 6px !important;
}
.stFileUploader:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button:hover,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button:hover {
  background: rgba(0,0,0,0.06) !important;
  color: #1a1a1e !important;
}
.stFileUploader:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button > *,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button > * {
  display: none !important;
}
.stFileUploader:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button::after,
[data-testid="stFileUploader"]:has(.stButton:nth-of-type(2)) .stButton:nth-of-type(2) > button::after {
  content: "\U0001F5D1" !important;
  display: inline !important;
  font-style: normal !important;
}
.stFileUploader label,
.stFileUploader p,
.stFileUploader span { color: #1a1a1e !important; font-size: 14px !important; }
.stFileUploader small { color: #6b7280 !important; font-size: 13px !important; }
.stSlider label { color: #1a1a1e !important; }

/* ---------- Buttons: light theme ---------- */
.stButton > button {
  font-size: 14px !important;
  font-weight: 600 !important;
  color: #1a1a1e !important;
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.15) !important;
  border-radius: 10px !important;
  padding: 10px 18px !important;
  transition: background 0.2s, border-color 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
  background: #f0f1f5 !important;
  border-color: rgba(0,0,0,0.25) !important;
}
.stButton > button:active {
  transform: scale(0.98);
}

/* ---------- Inputs (light theme) ---------- */
.stTextInput > div > div > input {
  border: 1px solid rgba(0,0,0,0.12) !important;
  border-radius: 12px !important;
  color: #1a1a1e !important;
  background: #ffffff !important;
}
.stFileUploader > div { border: none !important; }
[data-baseweb="slider"] { border: none !important; border-radius: 12px; }
</style>
""".replace("__BGM_B64__", _bgm_b64),
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "play_from" not in st.session_state:
    st.session_state.play_from = 0.0
if "video_duration" not in st.session_state:
    st.session_state.video_duration = 900.0  # 15 min fallback
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_jump" not in st.session_state:
    st.session_state.last_jump = None


def _msg_to_html(text: str) -> str:
    """Simple markdown to HTML: **bold** and `code`, with escaping."""
    import html
    s = html.escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s.replace("\n", "<br>")


def get_video_duration(video_bytes: bytes) -> float:
    """Get video duration in seconds using OpenCV."""
    try:
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            path = f.name
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        os.unlink(path)
        if frames and frames > 0:
            return frames / fps
    except Exception:
        pass
    return 900.0  # 15 min fallback when duration cannot be detected


def seek_to(timestamp_s: float, duration: float) -> float:
    return max(0.0, min(float(timestamp_s), duration))


# Default video: load check.mp4 from project dir if no video set yet
_default_video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.mp4")
if st.session_state.video_bytes is None and os.path.isfile(_default_video_path):
    with open(_default_video_path, "rb") as _f:
        st.session_state.video_bytes = _f.read()
    st.session_state.video_name = "check.mp4"
    st.session_state.video_duration = get_video_duration(st.session_state.video_bytes)


def _video_player_html(video_base64: str, mime: str, start_time: float) -> str:
    """Build a reference-style video player: semi-transparent bottom bar, purple progress, icon controls."""
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; }}
.vp-wrapper {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  color: #fff;
  height: 100%;
  min-height: 420px;
  position: relative;
}}
.vp-video {{
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: transparent;
  display: block;
}}
/* Timeline/controls overlay on top of video */
.vp-controls {{
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 2;
  background: rgba(0,0,0,0.35);
  padding: 10px 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.vp-row1 {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.vp-row2 {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.vp-center {{
  flex: 1;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}}
/* Purple progress bar with circular thumb */
.vp-progress-wrap {{
  flex: 1;
  height: 6px;
  background: rgba(255,255,255,0.2);
  border-radius: 3px;
  cursor: pointer;
  position: relative;
  min-width: 60px;
}}
.vp-progress-fill {{
  height: 100%;
  background: #a855f7;
  border-radius: 3px;
  width: 0%;
  transition: width 0.08s;
  position: relative;
}}
.vp-thumb {{
  position: absolute;
  right: -6px;
  top: 50%;
  transform: translateY(-50%);
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #a855f7;
  box-shadow: 0 0 0 2px rgba(255,255,255,0.3);
  pointer-events: none;
}}
.vp-time {{
  font-size: 12px;
  color: rgba(255,255,255,0.9);
  white-space: nowrap;
}}
/* Icon-style buttons */
.vp-icon {{
  width: 36px;
  height: 36px;
  border: none;
  background: transparent;
  color: #fff;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  transition: background 0.2s;
}}
.vp-icon:hover {{
  background: rgba(255,255,255,0.15);
}}
.vp-icon-play {{
  width: 40px;
  height: 40px;
  font-size: 20px;
}}
.vp-vol-wrap {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.vp-vol-wrap input {{
  width: 72px;
  height: 4px;
  accent-color: #a855f7;
  cursor: pointer;
}}
.vp-vol-icon {{ width: 20px; height: 20px; flex-shrink: 0; }}
#btnMute .vp-vol-icon.vol-off {{ display: none; }}
#btnMute.muted .vp-vol-icon.vol-on {{ display: none; }}
#btnMute.muted .vp-vol-icon.vol-off {{ display: block; }}
</style>
</head>
<body>
<div class="vp-wrapper">
  <video id="v" class="vp-video" preload="metadata">
    <source src="data:{mime};base64,{video_base64}" type="{mime}">
  </video>
  <div class="vp-controls">
    <div class="vp-row1">
      <button class="vp-icon vp-icon-play" id="btnPlay" title="Play/Pause">&#9654;</button>
      <div class="vp-center">
        <div class="vp-progress-wrap" id="progressWrap">
          <div class="vp-progress-fill" id="progressFill"><div class="vp-thumb"></div></div>
        </div>
        <span class="vp-time" id="timeDisplay">0:00 / 0:00</span>
      </div>
      <button class="vp-icon" id="btnSpeed" title="Speed">1x</button>
    </div>
    <div class="vp-row2">
      <button class="vp-icon" id="btnBack" title="Back 10s">&#8634;</button>
      <button class="vp-icon" id="btnFwd" title="Forward 10s">&#8635;</button>
      <div class="vp-vol-wrap">
        <button class="vp-icon" id="btnMute" title="Volume">
          <svg class="vp-vol-icon vol-on" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
          <svg class="vp-vol-icon vol-off" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>
        </button>
        <input type="range" id="vol" min="0" max="100" value="100" title="Volume">
      </div>
    </div>
  </div>
</div>
<script>
(function() {{
  const v = document.getElementById('v');
  const progressFill = document.getElementById('progressFill');
  const progressWrap = document.getElementById('progressWrap');
  const timeDisplay = document.getElementById('timeDisplay');
  const btnPlay = document.getElementById('btnPlay');
  const btnBack = document.getElementById('btnBack');
  const btnFwd = document.getElementById('btnFwd');
  const btnMute = document.getElementById('btnMute');
  const volInput = document.getElementById('vol');
  const btnSpeed = document.getElementById('btnSpeed');

  const startTime = {start_time};
  v.currentTime = startTime;

  const speeds = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
  let speedIdx = 2;

  function fmt(t) {{
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }}

  function update() {{
    const pct = v.duration ? (v.currentTime / v.duration) * 100 : 0;
    progressFill.style.width = pct + '%';
    timeDisplay.textContent = fmt(v.currentTime) + ' / ' + fmt(v.duration || 0);
    btnPlay.innerHTML = v.paused ? '&#9654;' : '&#10074;&#10074;';
  }}

  v.addEventListener('loadedmetadata', function() {{ v.currentTime = startTime; update(); }});
  v.addEventListener('timeupdate', update);
  v.addEventListener('play', update);
  v.addEventListener('pause', update);

  btnPlay.addEventListener('click', function() {{
    if (v.paused) v.play(); else v.pause();
    update();
  }});

  btnBack.addEventListener('click', function() {{
    v.currentTime = Math.max(0, v.currentTime - 10);
    update();
  }});

  btnFwd.addEventListener('click', function() {{
    v.currentTime = Math.min(v.duration || v.currentTime, v.currentTime + 10);
    update();
  }});

  progressWrap.addEventListener('click', function(e) {{
    const rect = progressWrap.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, x / rect.width));
    v.currentTime = (v.duration || 0) * pct;
    update();
  }});

  function updateMuteIcon() {{
    btnMute.classList.toggle("muted", v.volume === 0);
  }}

  volInput.addEventListener('input', function() {{
    v.volume = volInput.value / 100;
    updateMuteIcon();
  }});

  btnMute.addEventListener('click', function() {{
    if (v.volume > 0) {{ v.volume = 0; volInput.value = 0; }}
    else {{ v.volume = 1; volInput.value = 100; }}
    updateMuteIcon();
  }});

  btnSpeed.addEventListener('click', function() {{
    speedIdx = (speedIdx + 1) % speeds.length;
    v.playbackRate = speeds[speedIdx];
    btnSpeed.textContent = speeds[speedIdx] + 'x';
  }});

  v.addEventListener('loadeddata', function() {{ update(); }});

  document.addEventListener('keydown', function(e) {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') {{ e.preventDefault(); btnPlay.click(); }}
    if (e.code === 'ArrowLeft') {{ e.preventDefault(); btnBack.click(); }}
    if (e.code === 'ArrowRight') {{ e.preventDefault(); btnFwd.click(); }}
  }});

  update();
}})();
</script>
</body>
</html>
"""


# -----------------------------
# Model hook (replace later)
# -----------------------------
@dataclass
class ModelResult:
    answer: str
    timestamp_s: Optional[float] = None


def _parse_timestamp_hint(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*s\b", text.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"\b(?:(\d+):)?(\d{1,2}):(\d{2})\b", text)
    if m:
        hh = int(m.group(1)) if m.group(1) else 0
        mm = int(m.group(2))
        ss = int(m.group(3))
        return float(hh * 3600 + mm * 60 + ss)
    return None


def answer_question(video_bytes: bytes | None, question: str) -> ModelResult:
    if not video_bytes:
        return ModelResult("Upload a video first, then ask questions about it.", None)
    ts = _parse_timestamp_hint(question)
    if ts is not None:
        return ModelResult(f"Jumping to **{ts:.2f}s** (demo).", ts)
    return ModelResult(
        "Got it. Connect your model in `answer_question()` for real responses.",
        None,
    )


# -----------------------------
# Header (navigation bar: logo left, title + subtitle)
# -----------------------------
_logo_img = ""
if _logo_b64:
    _logo_img = f'<img src="data:image/png;base64,{_logo_b64}" alt="VISTA" class="nav-bar-logo" />'
st.markdown(
    f"""
    <div class="nav-bar-vibe">
      <div class="nav-bar-left">
        {_logo_img}
        <span class="nav-bar-title">VISTA</span>
      </div>
      <span class="nav-bar-subtitle"><strong>V</strong>ideo <strong>I</strong>ntelligence with <strong>S</strong>patio-<strong>T</strong>emporal <strong>A</strong>ugmented Retrieval for Egocentric Understanding</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.05, 0.95], gap="large")

# -----------------------------
# Left: Video
# -----------------------------
with left:
    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "m4v", "webm"],
        label_visibility="collapsed",
    )
    # Late-loading overrides so Browse files is yellow with black text (beats Streamlit defaults)
    st.markdown(
        """
        <style>
        [data-testid="stFileUploader"] [data-baseweb="button"]:first-of-type,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] > div > div:first-child button,
        [data-testid="stFileUploader"] section:first-of-type button {
            color: #000000 !important;
            background: linear-gradient(180deg, #fde047 0%, #eab308 50%, #ca8a04 100%) !important;
            background-color: #eab308 !important;
        }
        [data-testid="stFileUploader"] [data-baseweb="button"]:first-of-type span,
        [data-testid="stFileUploaderDropzone"] button *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button *,
        [data-testid="stFileUploader"] section:first-of-type button * {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if uploaded is not None:
        st.session_state.video_bytes = uploaded.getvalue()
        st.session_state.video_name = uploaded.name
        st.session_state.play_from = 0.0
        st.session_state.last_jump = None
        st.session_state.video_duration = get_video_duration(st.session_state.video_bytes)

    if not st.session_state.video_bytes:
        st.markdown(
            "<p style='color:#c8c8d4; font-size:14px; font-weight:500; line-height:1.5;'>Get started.</p>",
            unsafe_allow_html=True,
        )
    else:
        name = st.session_state.video_name or "video.mp4"
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "mp4"
        mime_map = {"mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm", "m4v": "video/x-m4v"}
        mime = mime_map.get(ext, "video/mp4")
        # Default video: serve from file path so /media/ 404 is avoided
        _check_mp4_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.mp4")
        if name == "check.mp4" and os.path.isfile(_check_mp4_path):
            st.video(_check_mp4_path, format="video/mp4")
        else:
            try:
                suffix = "." + ext if ext in ("mp4", "mov", "webm", "m4v") else ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(st.session_state.video_bytes)
                    tmp_path = tmp.name
                st.video(tmp_path, format=mime)
            except Exception:
                st.video(st.session_state.video_bytes, format=mime)

# -----------------------------
# Right: Chat
# -----------------------------
with right:
    chat_html = [
        "<div class='chat-panel'>",
        "<div class='chat-messages'>",
    ]
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        text = msg.get("text", "")
        ts = msg.get("timestamp_s", None)
        content = _msg_to_html(text)
        meta = f"<div class='chat-meta'>Jump to {float(ts):.1f}s</div>" if ts is not None else ""
        row_class = "user" if role == "user" else ""
        chat_html.append(
            f"<div class='chat-row {row_class}'>"
            f"<div class='chat-bubble'><p>{content}</p>{meta}</div>"
            f"</div>"
        )
    chat_html.append("</div></div>")
    st.markdown("\n".join(chat_html), unsafe_allow_html=True)

    # Chat input: transparent field + send button with upward arrow
    with st.form("chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([6, 1])
        with input_col:
            msg = st.text_input("Message", placeholder="Ask about the video...", label_visibility="collapsed", key="chat_msg")
        with btn_col:
            submitted = st.form_submit_button("\u2191")
    # Late-loading style to override Streamlit default (blue send button)
    st.markdown(
        """
        <style>
        [data-testid="stForm"] button,
        [data-testid="stForm"] .stButton button {
            background: linear-gradient(180deg, #5eb0ff 0%, #0077ed 50%, #0066d6 100%) !important;
            background-color: #0077ed !important;
            color: #fff !important;
            border: none !important;
        }
        [data-testid="stForm"] button:hover {
            background: linear-gradient(180deg, #70bcff 0%, #0088ff 50%, #0077ed 100%) !important;
            background-color: #0088ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if submitted and msg and msg.strip():
        user_q = msg.strip()
        st.session_state.messages.append({"role": "user", "text": user_q})

        with st.spinner("Thinking…"):
            time.sleep(0.2)
            res = answer_question(st.session_state.video_bytes, user_q)

        if res.timestamp_s is not None:
            st.session_state.play_from = seek_to(res.timestamp_s, st.session_state.video_duration if st.session_state.video_bytes else 900)
            st.session_state.last_jump = st.session_state.play_from

        st.session_state.messages.append(
            {"role": "assistant", "text": res.answer, "timestamp_s": res.timestamp_s}
        )
        st.rerun()

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="app-footer">© Team Cortex</div>',
    unsafe_allow_html=True,
)
