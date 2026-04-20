from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from firecastbot.config import Settings


@lru_cache(maxsize=8)
def _sentence_transformer_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbedderService:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings
        self.provider = (provider or settings.embedding_provider).strip().casefold()
        self.model = (model or settings.embedding_model).strip()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.provider == "sentence-transformers":
            return self._embed_with_sentence_transformers(texts)
        if self.provider == "openai":
            return self._embed_with_openai(texts)
        if self.provider == "gemini":
            return self._embed_with_gemini(texts)
        if self.provider == "openrouter":
            return self._embed_with_openrouter(texts)
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _embed_with_sentence_transformers(self, texts: list[str]) -> np.ndarray:
        model = _sentence_transformer_model(self.model)
        prepared_texts = [
            _prepare_embedding_text(text, model_name=self.model, is_query=False) for text in texts
        ]
        vectors = model.encode(
            prepared_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def _embed_with_gemini(self, texts: list[str]) -> np.ndarray:
        api_key = (
            (self.settings.gemini_api_key or "").strip()
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("FIRECASTBOT_GEMINI_API_KEY", "")
        )
        if not api_key:
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY in your environment.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "Install the google-generativeai package to enable Gemini embeddings."
            ) from exc
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_document",
        )
        vectors = np.asarray(result["embedding"], dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _embed_with_openrouter(self, texts: list[str]) -> np.ndarray:
        import requests as _requests

        api_key = (
            (self.settings.openrouter_api_key or "").strip()
            or os.environ.get("FIRECASTBOT_OPENROUTER_API_KEY", "")
            or os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "Missing OpenRouter API key. Set OPEN_ROUTER_API_KEY in your environment."
            )
        base_url = (self.settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        all_vectors: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = _requests.post(
                f"{base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            items = sorted(payload["data"], key=lambda x: x["index"])
            all_vectors.extend(item["embedding"] for item in items)
        vectors = np.asarray(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _embed_with_openai(self, texts: list[str]) -> np.ndarray:
        api_key = self.settings.require_api_key("openai")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the openai package to enable OpenAI embeddings.") from exc
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def prepare_query_texts(texts: list[str], model_name: str) -> list[str]:
    return [_prepare_embedding_text(text, model_name=model_name, is_query=True) for text in texts]


def _prepare_embedding_text(text: str, *, model_name: str, is_query: bool) -> str:
    normalized_model = model_name.strip().casefold()
    stripped = text.strip()
    if normalized_model.startswith("baai/bge-"):
        prefix = "Represent this sentence for searching relevant passages: "
        return f"{prefix}{stripped}" if is_query else stripped
    if normalized_model.startswith("intfloat/e5-"):
        prefix = "query: " if is_query else "passage: "
        return f"{prefix}{stripped}"
    return stripped
