from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

from firecastbot.config import Settings


class ChatClient(Protocol):
    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        ...

    def chat_completion_result(self, messages: list[dict[str, str]]) -> "ChatCompletionResult":
        ...


@dataclass(frozen=True)
class LLMRequest:
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    provider: str
    model: str
    response_id: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    response_metadata: dict | None = None


class LLMService:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.settings = settings
        self.request = LLMRequest(
            provider=(provider or settings.llm_provider).strip().casefold(),
            model=(model or settings.llm_model).strip(),
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
        )
        self._client = self._build_client()

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        return self._client.chat_completion(messages)

    def chat_completion_result(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        return self._client.chat_completion_result(messages)

    def _build_client(self) -> ChatClient:
        provider = self.request.provider
        if provider == "groq":
            return GroqChatClient(self.settings, self.request)
        if provider == "openai":
            return OpenAIChatClient(self.settings, self.request)
        if provider == "openrouter":
            return OpenRouterChatClient(self.settings, self.request)
        if provider == "anthropic":
            return AnthropicChatClient(self.settings, self.request)
        if provider == "gemini":
            return GeminiChatClient(self.settings, self.request)
        if provider == "xai":
            return XAIChatClient(self.settings, self.request)
        if provider == "ollama":
            return OllamaChatClient(self.settings, self.request)
        raise ValueError(f"Unsupported LLM provider: {provider}")


class GroqChatClient:
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        api_key = settings.require_api_key("groq")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the groq package to enable Groq models.") from exc
        self.client = Groq(api_key=api_key)
        self.request = request

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        return self.chat_completion_result(messages).content

    def chat_completion_result(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        full_response = ""
        stream = self.client.chat.completions.create(
            model=self.request.model,
            messages=messages,
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
            stream=True,
        )
        for chunk in stream:
            part = chunk.choices[0].delta.content
            if part:
                full_response += part
        return ChatCompletionResult(
            content=full_response,
            provider=self.request.provider,
            model=self.request.model,
        )


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        request: LLMRequest,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the openai package to enable this provider.") from exc
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.request = request

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        return self.chat_completion_result(messages).content

    def chat_completion_result(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        response = self.client.chat.completions.create(
            model=self.request.model,
            messages=messages,
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        completion_tokens_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None) if prompt_tokens_details else None
        reasoning_tokens = (
            getattr(completion_tokens_details, "reasoning_tokens", None)
            if completion_tokens_details
            else None
        )
        return ChatCompletionResult(
            content=choice.message.content or "",
            provider=self.request.provider,
            model=self.request.model,
            response_id=getattr(response, "id", "") or "",
            finish_reason=getattr(choice, "finish_reason", "") or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            response_metadata={
                "service_tier": getattr(response, "service_tier", None),
            },
        )


class OpenAIChatClient(OpenAICompatibleChatClient):
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        super().__init__(api_key=settings.require_api_key("openai"), request=request)


class XAIChatClient(OpenAICompatibleChatClient):
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        super().__init__(
            api_key=settings.require_api_key("xai"),
            request=request,
            base_url=settings.xai_base_url,
        )


class OpenRouterChatClient(OpenAICompatibleChatClient):
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        super().__init__(
            api_key=settings.require_api_key("openrouter"),
            request=request,
            base_url=settings.openrouter_base_url,
        )


class AnthropicChatClient:
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        api_key = settings.require_api_key("anthropic")
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the anthropic package to enable Claude models.") from exc
        self.client = Anthropic(api_key=api_key)
        self.request = request

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        return self.chat_completion_result(messages).content

    def chat_completion_result(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        system_messages = [message["content"] for message in messages if message["role"] == "system"]
        user_messages = [
            {
                "role": "assistant" if message["role"] == "assistant" else "user",
                "content": message["content"],
            }
            for message in messages
            if message["role"] != "system"
        ]
        response = self.client.messages.create(
            model=self.request.model,
            system="\n\n".join(system_messages),
            messages=user_messages,
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return ChatCompletionResult(
            content=content,
            provider=self.request.provider,
            model=self.request.model,
            response_id=getattr(response, "id", "") or "",
            finish_reason=getattr(response, "stop_reason", "") or "",
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            total_tokens=(
                (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
                if usage is not None
                else None
            ),
        )


class GeminiChatClient:
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        api_key = settings.require_api_key("gemini")
        self.request = request
        self._mode = ""
        self._client = None
        self._api_key = api_key
        self._configure_client()

    def _configure_client(self) -> None:
        try:
            from google import genai
        except ImportError:
            try:
                import google.generativeai as genai_legacy
            except ImportError as exc:  # pragma: no cover
                self._mode = "rest"
                self._client = None
                return
            genai_legacy.configure(api_key=self._api_key)
            self._mode = "legacy"
            self._client = genai_legacy.GenerativeModel(self.request.model)
            return
        self._mode = "sdk"
        self._client = genai.Client(api_key=self._api_key)

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        return self.chat_completion_result(messages).content

    def chat_completion_result(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        if self._mode == "legacy":
            prompt = _join_messages(messages)
            response = self._client.generate_content(prompt)
            return ChatCompletionResult(
                content=getattr(response, "text", "") or "",
                provider=self.request.provider,
                model=self.request.model,
            )

        if self._mode == "rest":
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.request.model}:generateContent",
                headers={
                    "x-goog-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": _join_messages(messages)}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": self.request.temperature,
                        "maxOutputTokens": self.request.max_tokens,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            candidates = payload.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini REST API returned no candidates.")
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            text = "".join(str(part.get("text") or "") for part in parts)
            if not text:
                raise RuntimeError("Gemini REST API returned no text content.")
            usage = payload.get("usageMetadata") or {}
            return ChatCompletionResult(
                content=text,
                provider=self.request.provider,
                model=self.request.model,
                response_id=str(payload.get("responseId") or ""),
                finish_reason=str(candidates[0].get("finishReason") or ""),
                prompt_tokens=usage.get("promptTokenCount"),
                completion_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount"),
                cached_tokens=usage.get("cachedContentTokenCount"),
                response_metadata={
                    "traffic_type": usage.get("trafficType"),
                },
            )

        prompt = _join_messages(messages)
        response = self._client.models.generate_content(
            model=self.request.model,
            contents=prompt,
        )
        usage = getattr(response, "usage_metadata", None)
        return ChatCompletionResult(
            content=getattr(response, "text", "") or "",
            provider=self.request.provider,
            model=self.request.model,
            response_id=str(getattr(response, "response_id", "") or ""),
            prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            total_tokens=getattr(usage, "total_token_count", None) if usage else None,
            cached_tokens=getattr(usage, "cached_content_token_count", None) if usage else None,
        )


class OllamaChatClient:
    def __init__(self, settings: Settings, request: LLMRequest) -> None:
        self.request = request
        self.base_url = settings.ollama_base_url.rstrip("/")

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        return self.chat_completion_result(messages).content

    def chat_completion_result(self, messages: list[dict[str, str]]) -> ChatCompletionResult:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.request.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.request.temperature,
                    "num_predict": self.request.max_tokens,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned a response without message content.")
        return ChatCompletionResult(
            content=content,
            provider=self.request.provider,
            model=self.request.model,
            finish_reason=str(payload.get("done_reason") or ""),
            response_metadata={
                "eval_count": payload.get("eval_count"),
                "prompt_eval_count": payload.get("prompt_eval_count"),
            },
        )


def _join_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )
