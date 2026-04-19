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

from firecastbot.config import get_settings
from firecastbot.prompts import build_summary_prompt
from firecastbot.services.llm_service import LLMService
from firecastbot.services.speech_service import SpeechService
from inferno_api.firecastbot_runtime import (
    build_grounded_prompt,
    build_runtime_incident_report,
    classify_query,
    load_doctrine_assets,
    retrieve_firecast_context,
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
    incident_embedding_provider: str = "sentence-transformers"
    incident_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    conversation: list[dict[str, str]] = field(default_factory=list)
    rolling_conversation_summary: str = ""
    summarized_message_count: int = 0
    latest_transcript: str = ""
    latest_query_classification: str = ""
    latest_retrieval_context: list[dict[str, Any]] = field(default_factory=list)


class FireCastBotManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm_service = LLMService(self.settings)
        doctrine_manifest = APPS_ROOT / "firecastbot" / "incident_response_docs" / "doctrine_retrieval_manifest.json"
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
            self.settings.embedding_provider,
            self.settings.embedding_model,
        )
        session.incident_profile = runtime_bundle["incident_profile"]
        session.incident_chunks = runtime_bundle["incident_chunks"]
        session.incident_embeddings = runtime_bundle["incident_embeddings"]
        session.incident_keyword_index = runtime_bundle["incident_keyword_index"]
        session.incident_embedding_provider = runtime_bundle["embedding_provider"]
        session.incident_embedding_model = runtime_bundle["embedding_model"]
        return {
            "documentsCount": len(session.incident_chunks),
            "incidentProfile": session.incident_profile,
        }

    def load_url(self, session_id: str, url: str) -> dict[str, Any]:
        raise ValueError("URL ingestion is no longer supported for FireCastBot incident sessions.")

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
        completion = self.llm_service.chat_completion_result(messages)
        reply = completion.content
        finish_reason = (completion.finish_reason or "").strip().casefold()

        continuation_attempts = 0
        while finish_reason in {"length", "max_tokens", "max_output_tokens"} and continuation_attempts < 2:
            continuation_messages = messages + [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you left off. Do not restart, do not repeat prior text, "
                        "and finish the answer cleanly."
                    ),
                },
            ]
            continuation = self.llm_service.chat_completion_result(continuation_messages)
            continuation_text = continuation.content.strip()
            if not continuation_text:
                break
            reply = f"{reply.rstrip()}\n{continuation_text}"
            finish_reason = (continuation.finish_reason or "").strip().casefold()
            continuation_attempts += 1

        session.conversation.append({"role": "user", "content": query})
        session.conversation.append({"role": "assistant", "content": reply})
        self._compact_conversation(session)
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
            "documentsCount": len(session.incident_chunks or []),
            "conversation": session.conversation,
            "latestTranscript": session.latest_transcript,
            "incidentProfile": session.incident_profile,
            "latestQueryClassification": session.latest_query_classification,
            "rollingConversationSummary": session.rolling_conversation_summary,
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
            conversation_summary=session.rolling_conversation_summary,
            recent_conversation=self._recent_conversation(session),
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a wildfire decision-support assistant grounded in incident facts and doctrine. "
                    "Answer direct incident questions directly and concisely before adding caveats or supporting detail. "
                    "Keep responses to a moderate length by default and avoid unnecessary verbosity or repetitive boilerplate. "
                    "Be spatial-context aware: tailor guidance to the incident's geography, terrain, and stated weather. "
                    "Be risk-context aware: use the incident's stated Overall Risk Level to calibrate urgency, caution, and safety emphasis. "
                    "You may mention likely regional conditions when location strongly implies them, but label those as inferred context, "
                    "not confirmed incident facts."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _recent_conversation(self, session: FireCastBotSession) -> list[dict[str, str]]:
        recent_message_limit = max(self.settings.chat_recent_turn_limit, 1) * 2
        return session.conversation[-recent_message_limit:]

    def _compact_conversation(self, session: FireCastBotSession) -> None:
        summarize_after_messages = max(self.settings.chat_summarize_after_turns, 1) * 2
        recent_message_limit = max(self.settings.chat_recent_turn_limit, 1) * 2

        if len(session.conversation) <= summarize_after_messages:
            return

        if len(session.conversation) <= recent_message_limit:
            return

        older_messages = session.conversation[:-recent_message_limit]
        if not older_messages:
            return

        messages = [
            {"role": "system", "content": "You create concise rolling conversation summaries."},
            {
                "role": "user",
                "content": build_summary_prompt(
                    older_messages,
                    existing_summary=session.rolling_conversation_summary,
                ),
            },
        ]
        session.rolling_conversation_summary = self.llm_service.chat_completion(messages)
        session.summarized_message_count += len(older_messages)
        session.conversation = session.conversation[-recent_message_limit:]

    def _retrieve_context(
        self,
        session: FireCastBotSession,
        query: str,
        query_class: str,
    ) -> list[dict[str, Any]]:
        return retrieve_firecast_context(
            query,
            query_class=query_class,
            retrieval_k=self.settings.retrieval_k,
            incident_profile=session.incident_profile,
            incident_chunks=session.incident_chunks,
            incident_embeddings=session.incident_embeddings,
            incident_keyword_index=session.incident_keyword_index,
            incident_embedding_provider=session.incident_embedding_provider,
            incident_embedding_model=session.incident_embedding_model,
            doctrine_store=self.doctrine_store,
        )


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
            "defaultLlmProvider": settings.llm_provider,
            "defaultLlmModel": settings.llm_model,
            "defaultEmbeddingProvider": settings.embedding_provider,
            "defaultEmbeddingModel": settings.embedding_model,
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


@firecastbot_bp.get("/sessions/<session_id>/debug-profile")
def firecastbot_debug_profile(session_id: str):
    try:
        session = get_manager().get_session(session_id)
        return jsonify(
            {
                "sessionId": session_id,
                "incidentProfile": session.incident_profile,
                "documentsCount": len(session.incident_chunks or []),
            }
        )
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
