from typing import Any

import streamlit as st


DEFAULT_SESSION_STATE: dict[str, Any] = {
    "loaded_docs": None,
    "document_db": None,
    "conversation": [],
    "query_input": "",
    "latest_output": "",
    "latest_output_audio": None,
    "latest_transcript": "",
    "chat_summary": "",
    "url_input": "",
    "speak_responses": False,
    "speech_to_text_provider_id": "",
    "text_to_speech_provider_id": "",
    "last_browser_speech_event_id": "",
}


def initialize_session_state() -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
