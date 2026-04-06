from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "browser_speech",
    path=str(Path(__file__).resolve().parent / "browser_speech_frontend"),
)


def render_browser_speech_component(
    *,
    enable_transcription: bool,
    speak_text: str,
    auto_speak: bool,
    key: str,
) -> dict[str, Any] | None:
    payload = _COMPONENT(
        enable_transcription=enable_transcription,
        speak_text=speak_text,
        auto_speak=auto_speak,
        key=key,
        default={
            "eventId": "",
            "transcript": "",
            "status": "",
            "supportsRecognition": True,
            "supportsSynthesis": True,
            "error": "",
        },
    )
    return payload if isinstance(payload, dict) else None
