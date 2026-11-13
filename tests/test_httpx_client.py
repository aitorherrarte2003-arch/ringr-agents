"""Cliente HTTP real, probado con un transporte simulado de httpx (sin red)."""

import json
from collections.abc import Callable
from datetime import date

import httpx
import pytest

from ringr_agents.agents import DebtAgent
from ringr_agents.http_client import HttpRequest, HttpTransportError
from ringr_agents.httpx_client import HttpxClient
from ringr_agents.models import ConversationModel, ParserModel
from ringr_agents.results import ActionOutcome

REQUEST = HttpRequest(
    method="POST",
    url="https://api.example.test/v1/thing",
    headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
    body={"request": "Cambio de tarifa"},
)


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], *, max_retries: int = 2
) -> tuple[HttpxClient, list[float]]:
    """HttpxClient con transporte simulado; devuelve también las esperas hechas."""
    sleeps: list[float] = []
    client = HttpxClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=max_retries,
        sleep=sleeps.append,
    )
    return client, sleeps


def replies(*statuses: int) -> Callable[[httpx.Request], httpx.Response]:
    """Handler que contesta con estos códigos, uno por intento."""
    pending = list(statuses)
    return lambda request: httpx.Response(pending.pop(0), json={"status": "ok"})


def test_sends_exactly_what_the_request_says() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"id": "abc"})

    client, _ = make_client(handler)
    response = client.send(REQUEST)

    assert response.status_code == 200
    assert response.body == {"id": "abc"}
    sent = received[0]
    assert sent.method == "POST"
    assert str(sent.url) == REQUEST.url
    assert sent.headers["Authorization"] == "Bearer t"
    assert json.loads(sent.content) == {"request": "Cambio de tarifa"}


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_transient_errors_are_retried_with_backoff(status: int) -> None:
    client, sleeps = make_client(replies(status, status, 200))

    assert client.send(REQUEST).status_code == 200
    assert sleeps == [0.5, 1.0]  # backoff exponencial


def test_gives_up_after_max_retries_and_returns_the_last_answer() -> None:
    client, sleeps = make_client(replies(500, 500, 503))

    assert client.send(REQUEST).status_code == 503
    assert len(sleeps) == 2


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    client, sleeps = make_client(replies(status))

    assert client.send(REQUEST).status_code == status
    assert sleeps == []


def test_network_errors_are_retried_then_raised() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectTimeout("tiempo agotado", request=request)

    client, sleeps = make_client(handler)

    with pytest.raises(HttpTransportError, match="ConnectTimeout"):
        client.send(REQUEST)
    assert len(attempts) == 3  # 1 intento + 2 reintentos
    assert sleeps == [0.5, 1.0]


def test_network_error_then_success() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("conexión rechazada", request=request)
        return httpx.Response(200)

    client, _ = make_client(handler)
    assert client.send(REQUEST).status_code == 200


def test_zero_retries_means_a_single_attempt() -> None:
    client, sleeps = make_client(replies(500), max_retries=0)

    assert client.send(REQUEST).status_code == 500
    assert sleeps == []


def test_negative_retries_are_rejected() -> None:
    with pytest.raises(ValueError, match="negativo"):
        HttpxClient(max_retries=-1)


@pytest.mark.parametrize(
    "response",
    [httpx.Response(200, text="OK"), httpx.Response(200, json=[1, 2]), httpx.Response(204)],
)
def test_non_object_bodies_become_empty(response: httpx.Response) -> None:
    client, _ = make_client(lambda request: response)
    assert client.send(REQUEST).body == {}


def test_works_as_a_context_manager() -> None:
    inner = httpx.Client(transport=httpx.MockTransport(replies(200)))
    with HttpxClient(inner) as client:
        client.send(REQUEST)
    assert inner.is_closed


def test_default_client_is_created_and_closed() -> None:
    with HttpxClient(timeout_seconds=3.0) as client:
        inner = client._client  # el httpx.Client que ha creado por dentro
        assert inner.timeout == httpx.Timeout(3.0)
        assert not inner.is_closed
    assert inner.is_closed


def test_a_corrupt_response_is_a_transport_error() -> None:
    def corrupt(request: httpx.Request) -> httpx.Response:
        raise httpx.DecodingError("gzip roto", request=request)

    client, _ = make_client(corrupt, max_retries=0)

    with pytest.raises(HttpTransportError, match="DecodingError"):
        client.send(REQUEST)


def test_plugs_into_an_agent_without_changing_it() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200)

    http, _ = make_client(handler)
    agent = DebtAgent(
        ConversationModel("Anotado."),
        ParserModel({"commitment_date": "2026-11-20", "committed_amount": 150}),
        http_client=http,
        today=lambda: date(2026, 11, 10),
    )

    assert agent.handle_turn("150 el 20").outcome is ActionOutcome.EXECUTED
    assert str(received[0].url) == "https://api.ringr.debt/v1/commitment"
    assert received[0].headers["Idempotency-Key"]
