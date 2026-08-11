"""Regression tests for the product-facing RAG request envelope."""

from gateway.platforms.api_server import _session_chat_kwrag


def test_rag_enabled_uses_original_user_text_and_optional_scope():
    request, error = _session_chat_kwrag(
        {
            "rag": {
                "enabled": True,
                "scope": {"sources": ["kakao"]},
            }
        },
        "Which room owns this task?",
    )

    assert error is None
    assert request == {
        "query": "Which room owns this task?",
        "sources": ["kakao"],
        "rooms": None,
    }


def test_rag_off_does_not_enter_retrieval():
    request, error = _session_chat_kwrag(
        {"rag": {"enabled": False}},
        "ordinary turn",
    )

    assert error is None
    assert request is None


def test_rag_requires_boolean_enabled():
    request, error = _session_chat_kwrag(
        {"rag": {"enabled": "true"}},
        "ordinary turn",
    )

    assert request is None
    assert error is not None
