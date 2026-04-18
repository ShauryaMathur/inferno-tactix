from __future__ import annotations

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
        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _embed_with_sentence_transformers(self, texts: list[str]) -> np.ndarray:
        model = _sentence_transformer_model(self.model)
        prepared_texts = [_prepare_embedding_text(text, model_name=self.model, is_query=False) for text in texts]
        vectors = model.encode(
            prepared_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

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
