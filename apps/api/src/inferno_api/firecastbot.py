from __future__ import annotations

import base64
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
APPS_ROOT = REPO_ROOT / "apps"

if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from chatwithme.config import get_settings
from chatwithme.prompts import build_summary_prompt
from chatwithme.services.groq_service import GroqService
from chatwithme.services.speech_service import SpeechService
from inferno_api.firecastbot_runtime import (
    build_grounded_prompt,
    build_runtime_incident_report,
    classify_query,
    load_doctrine_assets,
    merge_context,
    retrieve_chunks,
    retrieve_fact_records,
)

firecastbot_bp = Blueprint("firecastbot", __name__, url_prefix="/api/firecastbot")


class UploadedFileAdapter:
    def __init__(self, storage: Any, default_name: str) -> None:
        self._bytes = storage.read()
        self.name = getattr(storage, "filename", "") or default_name
        self.type = getattr(storage, "mimetype", "") or "application/octet-stream"

    def getbuffer(self) -> bytes:
        return self._bytes

    def getvalue(self) -> bytes:
        return self._bytes


@dataclass
class FireCastBotSession:
    incident_profile: dict[str, Any] | None = None
    incident_chunks: list[dict[str, Any]] = field(default_factory=list)
    incident_embeddings: Any = None
    incident_keyword_index: dict[str, Any] | None = None
    conversation: list[dict[str, str]] = field(default_factory=list)
    latest_transcript: str = ""
    chat_summary: str = ""
    latest_query_classification: str = ""
    latest_retrieval_context: list[dict[str, Any]] = field(default_factory=list)


class FireCastBotManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.groq_service = GroqService(self.settings)
        doctrine_manifest = APPS_ROOT / "chatwithme" / "incident_response_docs" / "doctrine_retrieval_manifest.json"
        self.doctrine_store = load_doctrine_assets(doctrine_manifest)
        self._sessions: dict[str, FireCastBotSession] = {}
        self._lock = threading.Lock()

    def create_session(self) -> tuple[str, FireCastBotSession]:
        session_id = uuid.uuid4().hex
        session = FireCastBotSession()
        with self._lock:
            self._sessions[session_id] = session
        return session_id, session

    def get_session(self, session_id: str) -> FireCastBotSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown FireCastBot session: {session_id}")
        return session

    def load_pdf(self, session_id: str, uploaded_file: Any) -> dict[str, Any]:
        session = self.get_session(session_id)
        runtime_bundle = build_runtime_incident_report(
            uploaded_file.getvalue(),
            getattr(uploaded_file, "name", "incident_report.pdf"),
            self.settings.embedding_model,
        )
        session.incident_profile = runtime_bundle["incident_profile"]
        session.incident_chunks = runtime_bundle["incident_chunks"]
        session.incident_embeddings = runtime_bundle["incident_embeddings"]
        session.incident_keyword_index = runtime_bundle["incident_keyword_index"]
        return {
            "documentsCount": len(session.incident_chunks),
            "incidentProfile": session.incident_profile,
        }

    def load_url(self, session_id: str, url: str) -> dict[str, Any]:
        raise ValueError("URL ingestion is no longer supported for FireCastBot incident sessions.")

    def build_vector_store(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if not session.incident_chunks:
            raise ValueError("Load an incident report before building retrieval state.")
        return {"documentsCount": len(session.incident_chunks), "elapsedTime": 0.0}

    def run_query(
        self,
        session_id: str,
        query: str,
        *,
        speak_responses: bool,
        text_to_speech_provider_id: str,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        query_class = classify_query(query)
        context_items = self._retrieve_context(session, query, query_class)
        messages = self._build_messages(session, query, query_class, context_items)
        reply = self.groq_service.chat_completion(messages)

        session.conversation.append({"role": "user", "content": query})
        session.conversation.append({"role": "assistant", "content": reply})
        session.latest_query_classification = query_class
        session.latest_retrieval_context = context_items

        audio_base64 = None
        audio_mime_type = None
        if speak_responses:
            speech_service = SpeechService(
                self.settings,
                speech_to_text_provider_id=self.settings.speech_to_text_provider,
                text_to_speech_provider_id=text_to_speech_provider_id,
            )
            if (
                speech_service.synthesis_available
                and speech_service.text_to_speech_provider.output_mode == "audio_bytes"
            ):
                audio_bytes = speech_service.synthesize(reply)
                audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                audio_mime_type = f"audio/{self.settings.text_to_speech_format}"

        return {
            "reply": reply,
            "conversation": session.conversation,
            "audioBase64": audio_base64,
            "audioMimeType": audio_mime_type,
            "queryClassification": query_class,
            "retrievalContext": context_items,
        }

    def summarize(self, session_id: str) -> str:
        session = self.get_session(session_id)
        messages = [
            {"role": "system", "content": "You are excellent at summarizing chats."},
            {"role": "user", "content": build_summary_prompt(session.conversation)},
        ]
        summary = self.groq_service.chat_completion(messages)
        session.chat_summary = summary
        return summary

    def transcribe(self, session_id: str, uploaded_audio: Any, provider_id: str) -> str:
        session = self.get_session(session_id)
        speech_service = SpeechService(
            self.settings,
            speech_to_text_provider_id=provider_id,
            text_to_speech_provider_id=self.settings.text_to_speech_provider,
        )
        if not speech_service.transcription_available:
            raise RuntimeError(
                speech_service.transcription_unavailable_reason
                or "Speech transcription is unavailable."
            )
        transcript = speech_service.transcribe(uploaded_audio)
        session.latest_transcript = transcript
        return transcript

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        return self._snapshot(session_id, session)

    def _snapshot(self, session_id: str, session: FireCastBotSession) -> dict[str, Any]:
        return {
            "sessionId": session_id,
            "documentsLoaded": bool(session.incident_chunks),
            "documentsCount": len(session.incident_chunks or []),
            "vectorStoreReady": bool(session.incident_chunks),
            "conversation": session.conversation,
            "latestTranscript": session.latest_transcript,
            "chatSummary": session.chat_summary,
            "incidentProfile": session.incident_profile,
            "latestQueryClassification": session.latest_query_classification,
        }

    def _build_messages(
        self,
        session: FireCastBotSession,
        query: str,
        query_class: str,
        context_items: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        prompt = build_grounded_prompt(
            query=query,
            query_class=query_class,
            incident_profile=session.incident_profile,
            context_items=context_items,
            conversation=session.conversation,
        )
        return [
            {"role": "system", "content": "You are a wildfire decision-support assistant grounded in incident facts and doctrine."},
            {"role": "user", "content": prompt},
        ]

    def _retrieve_context(
        self,
        session: FireCastBotSession,
        query: str,
        query_class: str,
    ) -> list[dict[str, Any]]:
        incident_fact_hits: list[dict[str, Any]] = []
        incident_chunk_hits: list[dict[str, Any]] = []
        doctrine_hits: list[dict[str, Any]] = []

        if session.incident_profile:
            incident_fact_hits = retrieve_fact_records(query, session.incident_profile)

        if session.incident_chunks and session.incident_keyword_index is not None and session.incident_embeddings is not None:
            incident_chunk_hits = retrieve_chunks(
                query,
                chunks=session.incident_chunks,
                embeddings=session.incident_embeddings,
                keyword_index=session.incident_keyword_index,
                model_name=self.settings.embedding_model,
                limit=self.settings.retrieval_k,
            )

        doctrine_hits = retrieve_chunks(
            query,
            chunks=self.doctrine_store["chunks"],
            embeddings=self.doctrine_store["embeddings"],
            keyword_index=self.doctrine_store["keyword_index"],
            model_name=self.settings.embedding_model,
            limit=self.settings.retrieval_k,
        )

        if query_class == "incident-fact":
            return merge_context(incident_fact_hits, incident_chunk_hits)
        if query_class == "doctrine":
            return merge_context(doctrine_hits)
        if query_class in {"incident+doctrine synthesis", "safety-critical"}:
            return merge_context(incident_fact_hits, incident_chunk_hits, doctrine_hits)
        if session.incident_profile:
            return merge_context(incident_fact_hits, incident_chunk_hits, doctrine_hits[:2])
        return merge_context(doctrine_hits)


_manager: FireCastBotManager | None = None
_manager_error: Exception | None = None


def get_manager() -> FireCastBotManager:
    global _manager, _manager_error
    if _manager is not None:
        return _manager
    if _manager_error is not None:
        raise _manager_error
    try:
        _manager = FireCastBotManager()
    except Exception as exc:  # pragma: no cover
        _manager_error = exc
        raise
    return _manager


def _provider_status(provider_id: str) -> dict[str, Any]:
    speech_service = SpeechService(
        get_manager().settings,
        speech_to_text_provider_id=provider_id,
        text_to_speech_provider_id=provider_id,
    )
    return {
        "id": provider_id,
        "label": SpeechService.provider_options()[provider_id],
        "transcriptionAvailable": speech_service.speech_to_text_provider.available,
        "transcriptionUnavailableReason": speech_service.speech_to_text_provider.unavailable_reason,
        "inputMode": speech_service.speech_to_text_provider.input_mode,
        "synthesisAvailable": speech_service.text_to_speech_provider.available,
        "synthesisUnavailableReason": speech_service.text_to_speech_provider.unavailable_reason,
        "outputMode": speech_service.text_to_speech_provider.output_mode,
    }


@firecastbot_bp.get("/config")
def firecastbot_config():
    settings = get_manager().settings
    provider_ids = list(SpeechService.provider_options().keys())
    return jsonify(
        {
            "defaultSpeechToTextProvider": settings.speech_to_text_provider,
            "defaultTextToSpeechProvider": settings.text_to_speech_provider,
            "providers": [_provider_status(provider_id) for provider_id in provider_ids],
        }
    )


@firecastbot_bp.post("/sessions")
def firecastbot_create_session():
    manager = get_manager()
    session_id, session = manager.create_session()
    return jsonify(manager._snapshot(session_id, session))


@firecastbot_bp.get("/sessions/<session_id>")
def firecastbot_get_session(session_id: str):
    try:
        return jsonify(get_manager().session_snapshot(session_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@firecastbot_bp.post("/documents/pdf")
def firecastbot_load_pdf():
    session_id = request.form.get("session_id", "")
    uploaded_file = request.files.get("file")
    if not session_id or uploaded_file is None:
        return jsonify({"error": "session_id and file are required."}), 400
    try:
        payload = get_manager().load_pdf(
            session_id,
            UploadedFileAdapter(uploaded_file, default_name="upload.pdf"),
        )
        return jsonify({**get_manager().session_snapshot(session_id), **payload})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@firecastbot_bp.post("/documents/url")
def firecastbot_load_url():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    url = str(data.get("url", "")).strip()
    if not session_id or not url:
        return jsonify({"error": "session_id and url are required."}), 400
    try:
        payload = get_manager().load_url(session_id, url)
        return jsonify({**get_manager().session_snapshot(session_id), **payload})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@firecastbot_bp.post("/vector-store")
def firecastbot_build_vector_store():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    if not session_id:
        return jsonify({"error": "session_id is required."}), 400
    try:
        payload = get_manager().build_vector_store(session_id)
        return jsonify({**get_manager().session_snapshot(session_id), **payload})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@firecastbot_bp.post("/query")
def firecastbot_query():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    query = str(data.get("query", "")).strip()
    speak_responses = bool(data.get("speak_responses", False))
    text_to_speech_provider_id = str(
        data.get("text_to_speech_provider_id", get_manager().settings.text_to_speech_provider)
    )
    if not session_id or not query:
        return jsonify({"error": "session_id and query are required."}), 400
    try:
        payload = get_manager().run_query(
            session_id,
            query,
            speak_responses=speak_responses,
            text_to_speech_provider_id=text_to_speech_provider_id,
        )
        return jsonify({**get_manager().session_snapshot(session_id), **payload})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@firecastbot_bp.post("/summary")
def firecastbot_summary():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", ""))
    if not session_id:
        return jsonify({"error": "session_id is required."}), 400
    try:
        summary = get_manager().summarize(session_id)
        return jsonify({**get_manager().session_snapshot(session_id), "summary": summary})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@firecastbot_bp.post("/transcribe")
def firecastbot_transcribe():
    session_id = request.form.get("session_id", "")
    provider_id = request.form.get("speech_to_text_provider_id", get_manager().settings.speech_to_text_provider)
    uploaded_audio = request.files.get("file")
    if not session_id or uploaded_audio is None:
        return jsonify({"error": "session_id and file are required."}), 400
    try:
        transcript = get_manager().transcribe(
            session_id,
            UploadedFileAdapter(uploaded_audio, default_name="recording.wav"),
            provider_id,
        )
        return jsonify({**get_manager().session_snapshot(session_id), "transcript": transcript})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
