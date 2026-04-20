from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from firecastbot.config import Settings

InputMode = Literal["upload", "browser", "none"]
OutputMode = Literal["audio_bytes", "browser", "none"]


class SpeechProvider(ABC):
    provider_id: str
    label: str
    input_mode: InputMode = "none"
    output_mode: OutputMode = "none"

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def unavailable_reason(self) -> str | None:
        raise NotImplementedError

    def transcribe(self, uploaded_audio: Any) -> str:
        raise RuntimeError(f"{self.label} does not support server-side transcription.")

    def synthesize(self, text: str) -> bytes | None:
        raise RuntimeError(f"{self.label} does not support server-side text-to-speech.")


class OpenAISpeechProvider(SpeechProvider):
    provider_id = "openai"
    label = "OpenAI"
    input_mode: InputMode = "upload"
    output_mode: OutputMode = "audio_bytes"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None
        self._init_error: str | None = None

        if not settings.openai_api_key:
            self._init_error = "Add OPENAI_API_KEY to enable OpenAI speech features."
            return

        try:
            from openai import OpenAI
        except ImportError:
            self._init_error = "Install the openai package to enable OpenAI speech features."
            return

        self._client = OpenAI(api_key=settings.openai_api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._init_error

    def transcribe(self, uploaded_audio: Any) -> str:
        client = self._require_client()
        audio_bytes = uploaded_audio.getvalue()
        mime_type = getattr(uploaded_audio, "type", "audio/wav")
        file_name = getattr(uploaded_audio, "name", "recording.wav")

        transcript = client.audio.transcriptions.create(
            model=self.settings.speech_to_text_model,
            file=(file_name, audio_bytes, mime_type),
        )
        text = getattr(transcript, "text", "").strip()

        if not text:
            raise RuntimeError("No speech was detected in the audio input.")

        return text

    def synthesize(self, text: str) -> bytes:
        client = self._require_client()
        normalized_text = " ".join(text.split())

        if not normalized_text:
            raise RuntimeError("Response text is empty.")

        response = client.audio.speech.create(
            model=self.settings.text_to_speech_model,
            voice=self.settings.text_to_speech_voice,
            response_format=self.settings.text_to_speech_format,
            input=normalized_text[:4096],
        )

        if hasattr(response, "read"):
            audio_bytes = response.read()
        elif hasattr(response, "content"):
            audio_bytes = response.content
        else:
            raise RuntimeError("Unexpected response from text-to-speech API.")

        if not audio_bytes:
            raise RuntimeError("Text-to-speech returned an empty audio response.")

        return audio_bytes

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(self._init_error or "Speech provider is not available.")
        return self._client


class GroqSpeechProvider(SpeechProvider):
    provider_id = "groq"
    label = "Groq Whisper"
    input_mode: InputMode = "upload"
    output_mode: OutputMode = "none"

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    WHISPER_MODEL = "whisper-large-v3-turbo"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None
        self._init_error: str | None = None

        api_key = (settings.groq_api_key or "").strip()
        if not api_key:
            self._init_error = "Add GROQ_API_KEY to enable Groq Whisper transcription."
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=self.GROQ_BASE_URL)
        except ImportError:
            self._init_error = "Install the openai package to enable Groq Whisper."

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._init_error

    def transcribe(self, uploaded_audio: Any) -> str:
        if self._client is None:
            raise RuntimeError(self._init_error or "Groq Whisper is not available.")
        audio_bytes = uploaded_audio.getvalue() if hasattr(uploaded_audio, "getvalue") else uploaded_audio.getbuffer()
        mime_type = getattr(uploaded_audio, "type", "audio/webm")
        file_name = getattr(uploaded_audio, "name", "recording.webm")
        transcript = self._client.audio.transcriptions.create(
            model=self.WHISPER_MODEL,
            file=(file_name, audio_bytes, mime_type),
        )
        text = getattr(transcript, "text", "").strip()
        if not text:
            raise RuntimeError("No speech was detected in the audio input.")
        return text


class BrowserSpeechProvider(SpeechProvider):
    provider_id = "browser"
    label = "Browser Web Speech API"
    input_mode: InputMode = "browser"
    output_mode: OutputMode = "browser"

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str | None:
        return None

    def synthesize(self, text: str) -> bytes | None:
        return None


class SpeechService:
    _PROVIDER_ORDER = ("groq", "browser")
    _PROVIDER_LABELS = {
        "groq": "Groq Whisper",
        "browser": "Browser",
    }

    def __init__(
        self,
        settings: Settings,
        *,
        speech_to_text_provider_id: str | None = None,
        text_to_speech_provider_id: str | None = None,
    ) -> None:
        self.settings = settings
        self._provider_cache: dict[str, SpeechProvider] = {}
        self.speech_to_text_provider = self._get_provider(
            speech_to_text_provider_id or settings.speech_to_text_provider
        )
        self.text_to_speech_provider = self._get_provider(
            text_to_speech_provider_id or settings.text_to_speech_provider
        )

    @classmethod
    def provider_options(cls) -> dict[str, str]:
        return {provider_id: cls._PROVIDER_LABELS[provider_id] for provider_id in cls._PROVIDER_ORDER}

    @property
    def transcription_available(self) -> bool:
        return self.speech_to_text_provider.available

    @property
    def transcription_unavailable_reason(self) -> str | None:
        return self.speech_to_text_provider.unavailable_reason

    @property
    def synthesis_available(self) -> bool:
        return self.text_to_speech_provider.available

    @property
    def synthesis_unavailable_reason(self) -> str | None:
        return self.text_to_speech_provider.unavailable_reason

    def transcribe(self, uploaded_audio: Any) -> str:
        return self.speech_to_text_provider.transcribe(uploaded_audio)

    def synthesize(self, text: str) -> bytes | None:
        return self.text_to_speech_provider.synthesize(text)

    def _get_provider(self, provider_id: str) -> SpeechProvider:
        normalized_provider_id = provider_id.lower()
        if normalized_provider_id not in self._provider_cache:
            if normalized_provider_id == "groq":
                self._provider_cache[normalized_provider_id] = GroqSpeechProvider(self.settings)
            elif normalized_provider_id == "openai":
                self._provider_cache[normalized_provider_id] = OpenAISpeechProvider(self.settings)
            elif normalized_provider_id == "browser":
                self._provider_cache[normalized_provider_id] = BrowserSpeechProvider()
            else:
                raise ValueError(f"Unsupported speech provider: {provider_id}")
        return self._provider_cache[normalized_provider_id]
