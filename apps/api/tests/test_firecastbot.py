"""
Tests for inferno_api.firecastbot — session lifecycle, route validation,
size limits, and thread-safety primitives.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# FireCastBotSession — pure dataclass tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestFireCastBotSession:
    def _make_session(self):
        from inferno_api.firecastbot import FireCastBotSession

        return FireCastBotSession()

    def test_lock_is_rlock(self):
        session = self._make_session()
        assert isinstance(session._lock, type(threading.RLock()))

    def test_last_accessed_is_recent(self):
        before = time.time()
        session = self._make_session()
        after = time.time()
        assert before <= session.last_accessed <= after

    def test_default_fields(self):
        session = self._make_session()
        assert session.incident_profile is None
        assert session.incident_chunks == []
        assert session.conversation == []
        assert session.rolling_conversation_summary == ""
        assert session.summarized_message_count == 0
        assert session.latest_transcript == ""

    def test_two_sessions_have_independent_locks(self):
        from inferno_api.firecastbot import FireCastBotSession

        s1, s2 = FireCastBotSession(), FireCastBotSession()
        assert s1._lock is not s2._lock

    def test_two_sessions_have_independent_conversations(self):
        from inferno_api.firecastbot import FireCastBotSession

        s1, s2 = FireCastBotSession(), FireCastBotSession()
        s1.conversation.append({"role": "user", "content": "hello"})
        assert s2.conversation == []


# ---------------------------------------------------------------------------
# FireCastBotManager — session lifecycle (heavy deps mocked)
# ---------------------------------------------------------------------------


class TestFireCastBotManager:
    @pytest.fixture(autouse=True)
    def _patch_heavy(self):
        with (
            patch(
                "inferno_api.firecastbot.get_settings",
                return_value=MagicMock(
                    chat_recent_turn_limit=6,
                    chat_summarize_after_turns=8,
                    speech_to_text_provider="browser",
                    text_to_speech_provider="browser",
                ),
            ),
            patch("inferno_api.firecastbot.LLMService", return_value=MagicMock()),
            patch("inferno_api.firecastbot.load_doctrine_assets", return_value={}),
        ):
            yield

    def _make_manager(self):
        from inferno_api.firecastbot import FireCastBotManager

        return FireCastBotManager()

    def test_create_session_returns_id_and_session(self):
        manager = self._make_manager()
        session_id, session = manager.create_session()
        assert isinstance(session_id, str) and len(session_id) == 32  # uuid4().hex
        from inferno_api.firecastbot import FireCastBotSession

        assert isinstance(session, FireCastBotSession)

    def test_get_session_returns_correct_session(self):
        manager = self._make_manager()
        session_id, session = manager.create_session()
        assert manager.get_session(session_id) is session

    def test_get_session_updates_last_accessed(self):
        manager = self._make_manager()
        session_id, session = manager.create_session()
        old_ts = session.last_accessed
        time.sleep(0.01)
        manager.get_session(session_id)
        assert session.last_accessed >= old_ts

    def test_get_session_raises_for_unknown_id(self):
        manager = self._make_manager()
        with pytest.raises(KeyError, match="Unknown FireCastBot session"):
            manager.get_session("nonexistent")

    def test_evict_expired_sessions(self):
        from inferno_api.firecastbot import SESSION_TTL_SECONDS

        manager = self._make_manager()
        session_id, session = manager.create_session()
        # Backdate last_accessed beyond the TTL
        session.last_accessed = time.time() - SESSION_TTL_SECONDS - 1
        manager._evict_expired_sessions()
        with pytest.raises(KeyError):
            manager.get_session(session_id)

    def test_evict_does_not_remove_active_sessions(self):
        manager = self._make_manager()
        session_id, _ = manager.create_session()
        manager._evict_expired_sessions()
        # Should still be accessible
        manager.get_session(session_id)

    def test_concurrent_session_creation_is_safe(self):
        manager = self._make_manager()
        results = []
        errors = []

        def create():
            try:
                sid, _ = manager.create_session()
                results.append(sid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=create) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(results)) == 20  # all IDs unique

    def test_snapshot_copies_conversation(self):
        manager = self._make_manager()
        session_id, session = manager.create_session()
        session.conversation.append({"role": "user", "content": "hi"})
        snap = manager._snapshot(session_id, session)
        # Mutating the snapshot must not affect session
        snap["conversation"].clear()
        assert session.conversation != []

    def test_speech_service_is_cached(self):
        manager = self._make_manager()
        with patch("inferno_api.firecastbot.SpeechService") as MockSpeechService:
            MockSpeechService.return_value = MagicMock()
            _ = manager._get_speech_service("groq", "browser")
            _ = manager._get_speech_service("groq", "browser")
            # SpeechService constructor called only once for the same key
            assert MockSpeechService.call_count == 1

    def test_speech_service_different_keys_create_separate_instances(self):
        manager = self._make_manager()
        with patch("inferno_api.firecastbot.SpeechService") as MockSpeechService:
            MockSpeechService.side_effect = lambda *a, **kw: MagicMock()
            _ = manager._get_speech_service("groq", "browser")
            _ = manager._get_speech_service("browser", "browser")
            assert MockSpeechService.call_count == 2

    def test_recent_message_limit_property(self):
        manager = self._make_manager()
        # chat_recent_turn_limit = 6 → limit = 12
        assert manager._recent_message_limit == 12

    def test_summarize_after_messages_property(self):
        manager = self._make_manager()
        # chat_summarize_after_turns = 8 → limit = 16
        assert manager._summarize_after_messages == 16


# ---------------------------------------------------------------------------
# Route validation — uses Flask test client (manager fully mocked)
# ---------------------------------------------------------------------------


class TestRouteInputValidation:
    def test_create_session_returns_200(self, client, mock_manager):
        mock_manager.create_session.return_value = ("abc123", MagicMock())
        resp = client.post("/api/firecastbot/sessions")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "sessionId" in data

    def test_get_unknown_session_returns_404(self, client, mock_manager):
        mock_manager.session_snapshot.side_effect = KeyError("Unknown FireCastBot session: xyz")
        resp = client.get("/api/firecastbot/sessions/xyz")
        assert resp.status_code == 404
        assert "error" in json.loads(resp.data)

    def test_load_pdf_missing_session_id_returns_400(self, client):
        data = {"file": (io.BytesIO(b"%PDF-1.4 stub"), "test.pdf")}
        resp = client.post(
            "/api/firecastbot/documents/pdf", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 400
        assert "session_id" in json.loads(resp.data)["error"]

    def test_load_pdf_missing_file_returns_400(self, client):
        resp = client.post(
            "/api/firecastbot/documents/pdf",
            data={"session_id": "sess1"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_load_pdf_oversized_returns_413(self, client):
        big_pdf = io.BytesIO(b"X" * (21 * 1024 * 1024))  # 21 MB > 20 MB limit
        resp = client.post(
            "/api/firecastbot/documents/pdf",
            data={"session_id": "sess1", "file": (big_pdf, "big.pdf")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413

    def test_load_preset_missing_fields_returns_400(self, client):
        resp = client.post("/api/firecastbot/documents/preset", json={})
        assert resp.status_code == 400

    def test_load_preset_unknown_preset_returns_400(self, client, mock_manager):
        mock_manager.load_preset.side_effect = ValueError("Unknown preset: bogus")
        resp = client.post(
            "/api/firecastbot/documents/preset", json={"session_id": "s1", "preset_id": "bogus"}
        )
        assert resp.status_code == 400

    def test_query_missing_query_returns_400(self, client):
        resp = client.post("/api/firecastbot/query", json={"session_id": "s1"})
        assert resp.status_code == 400

    def test_query_missing_session_id_returns_400(self, client):
        resp = client.post("/api/firecastbot/query", json={"query": "what is the wind speed?"})
        assert resp.status_code == 400

    def test_transcribe_missing_file_returns_400(self, client):
        resp = client.post(
            "/api/firecastbot/transcribe",
            data={"session_id": "s1"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_transcribe_oversized_audio_returns_413(self, client):
        big_audio = io.BytesIO(b"X" * (26 * 1024 * 1024))  # 26 MB > 25 MB limit
        resp = client.post(
            "/api/firecastbot/transcribe",
            data={"session_id": "s1", "file": (big_audio, "rec.webm")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413

    def test_preset_pdf_unknown_returns_404(self, client):
        resp = client.get("/api/firecastbot/presets/nonexistent/pdf")
        assert resp.status_code == 404

    def test_config_endpoint_returns_200(self, client, mock_manager):
        with patch("inferno_api.firecastbot.SpeechService") as MockSS:
            instance = MagicMock()
            instance.speech_to_text_provider.available = True
            instance.speech_to_text_provider.unavailable_reason = None
            instance.speech_to_text_provider.input_mode = "browser"
            instance.text_to_speech_provider.available = True
            instance.text_to_speech_provider.unavailable_reason = None
            instance.text_to_speech_provider.output_mode = "browser"
            MockSS.return_value = instance
            MockSS.provider_options.return_value = {"browser": "Browser"}
            resp = client.get("/api/firecastbot/config")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "providers" in data
        assert "presets" in data


# ---------------------------------------------------------------------------
# get_manager singleton — thread-safety test
# ---------------------------------------------------------------------------


class TestGetManagerSingleton:
    def test_concurrent_calls_return_same_instance(self):
        """Double-checked locking must ensure only one FireCastBotManager is created."""
        import inferno_api.firecastbot as fb_module

        created = []

        original_cls = None

        def tracking_init(self_inner):
            created.append(1)
            # Set minimal required attributes so the manager doesn't crash
            self_inner._sessions = {}
            self_inner._lock = threading.Lock()
            self_inner._speech_service_cache = {}
            self_inner.settings = MagicMock(
                chat_recent_turn_limit=6,
                chat_summarize_after_turns=8,
                speech_to_text_provider="browser",
                text_to_speech_provider="browser",
            )
            self_inner.llm_service = MagicMock()
            self_inner.doctrine_store = {}
            cleanup = threading.Thread(target=lambda: None, daemon=True)
            cleanup.start()

        # Reset module-level singleton state
        old_manager = fb_module._manager
        old_error = fb_module._manager_error
        fb_module._manager = None
        fb_module._manager_error = None

        try:
            with patch.object(fb_module.FireCastBotManager, "__init__", tracking_init):
                results = []
                errors_list = []

                def call_get_manager():
                    try:
                        results.append(fb_module.get_manager())
                    except Exception as exc:
                        errors_list.append(exc)

                threads = [threading.Thread(target=call_get_manager) for _ in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                assert not errors_list
                # All threads should receive the same instance
                assert len(set(id(r) for r in results)) == 1
                # Manager __init__ should have been called exactly once
                assert len(created) == 1
        finally:
            fb_module._manager = old_manager
            fb_module._manager_error = old_error


# ---------------------------------------------------------------------------
# _SlidingWindowRateLimiter — unit tests
# ---------------------------------------------------------------------------


class TestSlidingWindowRateLimiter:
    def _make(self, max_requests=3, window=1.0):
        from inferno_api.firecastbot import _SlidingWindowRateLimiter

        return _SlidingWindowRateLimiter(max_requests, window)

    def test_allows_up_to_limit(self):
        limiter = self._make(max_requests=3)
        for _ in range(3):
            allowed, _ = limiter.is_allowed("ip-a")
            assert allowed

    def test_blocks_over_limit(self):
        limiter = self._make(max_requests=3)
        for _ in range(3):
            limiter.is_allowed("ip-a")
        allowed, retry_after = limiter.is_allowed("ip-a")
        assert not allowed
        assert retry_after > 0

    def test_independent_keys_do_not_interfere(self):
        limiter = self._make(max_requests=1)
        limiter.is_allowed("ip-a")
        # ip-a is now blocked, but ip-b should still be allowed
        allowed, _ = limiter.is_allowed("ip-b")
        assert allowed

    def test_window_expiry_allows_new_requests(self):
        limiter = self._make(max_requests=2, window=0.05)
        limiter.is_allowed("ip-a")
        limiter.is_allowed("ip-a")
        # Blocked now
        assert not limiter.is_allowed("ip-a")[0]
        # After the window expires the slot should be free again
        time.sleep(0.06)
        allowed, _ = limiter.is_allowed("ip-a")
        assert allowed

    def test_retry_after_is_positive_when_blocked(self):
        limiter = self._make(max_requests=1, window=1.0)
        limiter.is_allowed("ip-a")
        _, retry_after = limiter.is_allowed("ip-a")
        assert 0 < retry_after <= 1.0

    def test_cleanup_removes_idle_keys(self):
        limiter = self._make(max_requests=3, window=0.01)
        limiter.is_allowed("ip-stale")
        time.sleep(0.02)
        # Force cleanup by backdating the last_cleanup timestamp
        limiter._last_cleanup = time.monotonic() - limiter._CLEANUP_INTERVAL - 1
        limiter.is_allowed("ip-trigger-cleanup")
        assert "ip-stale" not in limiter._windows

    def test_thread_safety_under_concurrent_load(self):
        limiter = self._make(max_requests=100, window=5.0)
        results = []
        errors = []

        def hit():
            try:
                results.append(limiter.is_allowed("shared-ip")[0])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=hit) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 50


# ---------------------------------------------------------------------------
# Rate limiting — HTTP integration tests
# ---------------------------------------------------------------------------


class TestRateLimitingRoutes:
    """Verify that rate-limited endpoints return 429 once the limit is hit."""

    def _hit(self, client, n: int, **kwargs):
        responses = []
        for _ in range(n):
            responses.append(client.post("/api/firecastbot/sessions", **kwargs))
        return responses

    def test_requests_within_limit_succeed(self, client, mock_manager):
        mock_manager.create_session.return_value = ("s1", MagicMock())
        resps = self._hit(client, 3)
        assert all(r.status_code == 200 for r in resps)

    def test_fourth_request_is_rate_limited(self, client, mock_manager):
        mock_manager.create_session.return_value = ("s1", MagicMock())
        self._hit(client, 3)
        resp = client.post("/api/firecastbot/sessions")
        assert resp.status_code == 429
        body = json.loads(resp.data)
        assert "error" in body
        assert "retryAfter" in body

    def test_429_includes_retry_after_header(self, client, mock_manager):
        mock_manager.create_session.return_value = ("s1", MagicMock())
        self._hit(client, 3)
        resp = client.post("/api/firecastbot/sessions")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_different_ips_have_independent_limits(self, client, mock_manager):
        mock_manager.create_session.return_value = ("s1", MagicMock())
        # Exhaust the limit for ip-a
        for _ in range(3):
            client.post("/api/firecastbot/sessions", headers={"X-Forwarded-For": "1.2.3.4"})
        # ip-b should still be allowed
        resp = client.post("/api/firecastbot/sessions", headers={"X-Forwarded-For": "9.8.7.6"})
        assert resp.status_code == 200

    def test_query_endpoint_is_rate_limited(self, client):
        for _ in range(3):
            client.post("/api/firecastbot/query", json={"session_id": "s", "query": "hi"})
        resp = client.post("/api/firecastbot/query", json={"session_id": "s", "query": "hi"})
        assert resp.status_code == 429

    def test_transcribe_endpoint_is_rate_limited(self, client):
        audio = io.BytesIO(b"fake-audio")
        for _ in range(3):
            client.post(
                "/api/firecastbot/transcribe",
                data={"session_id": "s", "file": (io.BytesIO(b"a"), "r.webm")},
                content_type="multipart/form-data",
            )
        resp = client.post(
            "/api/firecastbot/transcribe",
            data={"session_id": "s", "file": (audio, "r.webm")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 429

    def test_read_endpoints_are_not_rate_limited(self, client, mock_manager):
        """GET /config and GET /sessions/<id> are exempt — no @rate_limit applied."""
        from collections import deque
        from inferno_api import firecastbot as fb_module

        # Saturate the IP bucket so any rate-limited POST would return 429
        fb_module._ip_limiter._windows["127.0.0.1"] = deque([time.monotonic()] * 100)
        # GET /config should still work
        with patch("inferno_api.firecastbot.SpeechService") as MockSS:
            inst = MagicMock()
            inst.speech_to_text_provider.available = True
            inst.speech_to_text_provider.unavailable_reason = None
            inst.speech_to_text_provider.input_mode = "browser"
            inst.text_to_speech_provider.available = True
            inst.text_to_speech_provider.unavailable_reason = None
            inst.text_to_speech_provider.output_mode = "browser"
            MockSS.return_value = inst
            MockSS.provider_options.return_value = {"browser": "Browser"}
            resp = client.get("/api/firecastbot/config")
        assert resp.status_code == 200
