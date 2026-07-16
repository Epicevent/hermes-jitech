"""Regression tests — stale-stream abort must reach native adapter clients.

Root cause of the oc17 19.5-minute gemini stall (2026-07-15): the
stale-stream / interrupt abort path finds sockets via
``_iter_pool_sockets``, which only knew the OpenAI SDK client shape
(``client._client``).  Hermes's native adapter facades
(``GeminiNativeClient``, ``GeminiCloudCodeClient``) keep their
``httpx.Client`` at ``client._http`` instead, so every abort was a
silent no-op (``tcp_force_closed=0``): the blocked worker never
unwound and the poll loop repeated ineffective kills forever.

These tests pin:

1. ``_iter_pool_sockets`` discovers the pool behind ``_http``.
2. The real adapter classes actually store their httpx client at
   ``_http`` (attribute-name pin — a rename must break this suite).
3. The OpenAI SDK shape (``_client``) keeps working.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import socket as _socket


class _FakeSocket:
    """Records shutdown/close calls without touching real FDs."""

    def __init__(self):
        self.shutdown_calls = []
        self.close_calls = 0

    def shutdown(self, how):
        self.shutdown_calls.append(how)

    def close(self):  # pragma: no cover — must not run (#29507)
        self.close_calls += 1


def _fake_httpx_shape(sock):
    """Mimic the httpcore-1 layout below an ``httpx.Client``."""
    stream = SimpleNamespace(_sock=sock)
    http11 = SimpleNamespace(_network_stream=stream)
    pool_entry = SimpleNamespace(_connection=http11)
    pool = SimpleNamespace(_connections=[pool_entry])
    transport = SimpleNamespace(_pool=pool)
    return SimpleNamespace(_transport=transport)


def test_iter_pool_sockets_discovers_http_attr_shape():
    """A client that keeps httpx at ``_http`` (native adapters) must be swept."""
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    sock = _FakeSocket()
    client = SimpleNamespace(_http=_fake_httpx_shape(sock))

    n = force_close_tcp_sockets(client)

    assert n == 1, (
        "abort must find sockets behind client._http — a 0 here is the "
        "silent no-op kill that let the oc17 gemini stream hang for 19.5 min"
    )
    assert sock.shutdown_calls == [_socket.SHUT_RDWR]
    assert sock.close_calls == 0


def test_openai_sdk_shape_still_swept():
    """Regression guard: the original ``_client`` shape keeps working."""
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    sock = _FakeSocket()
    client = SimpleNamespace(_client=_fake_httpx_shape(sock))

    assert force_close_tcp_sockets(client) == 1
    assert sock.shutdown_calls == [_socket.SHUT_RDWR]


def test_client_attr_wins_when_both_present():
    """``_client`` (OpenAI SDK) takes precedence; ``_http`` is the fallback."""
    from agent.agent_runtime_helpers import force_close_tcp_sockets

    sdk_sock = _FakeSocket()
    other_sock = _FakeSocket()
    client = SimpleNamespace(
        _client=_fake_httpx_shape(sdk_sock),
        _http=_fake_httpx_shape(other_sock),
    )

    assert force_close_tcp_sockets(client) == 1
    assert sdk_sock.shutdown_calls == [_socket.SHUT_RDWR]
    assert other_sock.shutdown_calls == []


def test_gemini_native_client_sockets_are_reachable():
    """End-to-end pin on the real class: GeminiNativeClient keeps its
    httpx client at ``_http`` and the abort sweep reaches its pool."""
    from agent.agent_runtime_helpers import force_close_tcp_sockets
    from agent.gemini_native_adapter import GeminiNativeClient

    sock = _FakeSocket()
    client = GeminiNativeClient(
        api_key="test-key",
        http_client=_fake_httpx_shape(sock),  # constructor stores at _http
    )

    assert client._http is not None, "attribute-name pin: _http must exist"
    n = force_close_tcp_sockets(client)

    assert n == 1, (
        "force_close_tcp_sockets must shut down GeminiNativeClient's pool "
        "sockets; 0 means the stale-stream watchdog is toothless for gemini"
    )
    assert sock.shutdown_calls == [_socket.SHUT_RDWR]


def test_gemini_cloudcode_client_sockets_are_reachable():
    """Same pin for the Cloud Code Assist adapter (gemini-cli provider)."""
    from agent.agent_runtime_helpers import force_close_tcp_sockets
    from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

    client = GeminiCloudCodeClient()
    try:
        # Attribute-name pin against the real constructor...
        assert isinstance(client._http, httpx.Client)
    finally:
        client._http.close()

    # ...then swap in the fake pool shape and confirm the sweep reaches it.
    sock = _FakeSocket()
    client._http = _fake_httpx_shape(sock)

    assert force_close_tcp_sockets(client) == 1
    assert sock.shutdown_calls == [_socket.SHUT_RDWR]
