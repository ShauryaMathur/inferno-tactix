"""
Shared pytest fixtures for inferno_api tests.

Heavy external dependencies (LLMService, load_doctrine_assets) are patched at
the module level so tests run without model files or API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure both src/ (for inferno_api) and apps/ (for firecastbot.*) are importable
API_SRC = Path(__file__).resolve().parents[1] / "src"
APPS_ROOT = Path(__file__).resolve().parents[3] / "apps"
for p in (str(API_SRC), str(APPS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Clear in-memory rate-limiter state before every test so tests are isolated."""
    from inferno_api import firecastbot as fb_module

    fb_module._ip_limiter._windows.clear()
    fb_module._session_limiter._windows.clear()
    yield
    fb_module._ip_limiter._windows.clear()
    fb_module._session_limiter._windows.clear()


@pytest.fixture()
def mock_manager():
    """A MagicMock that quacks like a FireCastBotManager."""
    manager = MagicMock()
    # Ensure settings attributes are plain strings so Flask can JSON-serialise them
    manager.settings.llm_provider = "openrouter"
    manager.settings.llm_model = "test-model"
    manager.settings.embedding_provider = "sentence-transformers"
    manager.settings.embedding_model = "test-emb"
    manager.settings.speech_to_text_provider = "browser"
    manager.settings.text_to_speech_provider = "browser"
    # Default snapshot returned by session operations
    manager.session_snapshot.return_value = {
        "sessionId": "testsession",
        "documentsCount": 0,
        "conversation": [],
        "latestTranscript": "",
        "incidentProfile": None,
        "latestQueryClassification": "",
        "rollingConversationSummary": "",
    }
    manager._snapshot.return_value = manager.session_snapshot.return_value
    return manager


@pytest.fixture()
def app(mock_manager):
    """Flask test application with get_manager() stubbed out."""
    with (
        patch("inferno_api.firecastbot.get_manager", return_value=mock_manager),
        patch("inferno_api.firecastbot.load_doctrine_assets", return_value={}),
        patch("inferno_api.firecastbot.LLMService", return_value=MagicMock()),
    ):
        from inferno_api.app_firecastbot import app as flask_app

        flask_app.config["TESTING"] = True
        yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()
