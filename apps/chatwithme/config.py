import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    openai_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 1024
    groq_temperature: float = 1.3
    retrieval_k: int = 3
    pdf_chunk_size: int = 2000
    pdf_chunk_overlap: int = 100
    web_chunk_size: int = 200
    web_chunk_overlap: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    speech_to_text_provider: str = "browser"
    text_to_speech_provider: str = "browser"
    speech_to_text_model: str = "gpt-4o-mini-transcribe"
    text_to_speech_model: str = "gpt-4o-mini-tts"
    text_to_speech_voice: str = "alloy"
    text_to_speech_format: str = "mp3"


def get_settings() -> Settings:
    package_root = Path(__file__).resolve().parent
    apps_root = package_root.parent
    repo_root = apps_root.parent

    for env_path in (
        repo_root / ".env",
        repo_root / ".env.local",
        apps_root / ".env",
        apps_root / ".env.local",
        package_root / ".env",
        package_root / ".env.local",
    ):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)

    api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    speech_to_text_provider = os.getenv("SPEECH_TO_TEXT_PROVIDER", "browser")
    text_to_speech_provider = os.getenv("TEXT_TO_SPEECH_PROVIDER", "browser")

    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it to your environment or .env file.")

    return Settings(
        groq_api_key=api_key,
        openai_api_key=openai_api_key,
        speech_to_text_provider=speech_to_text_provider,
        text_to_speech_provider=text_to_speech_provider,
    )
