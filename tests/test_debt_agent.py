from collections.abc import Iterator

from ringr_agents.agents import DebtAgent, DebtCommitment
from ringr_agents.conversation import Conversation
from ringr_agents.http_client import (
    HttpRequest,
    HttpResponse,
    HttpTransportError,
    JsonScalar,
    SimulatedHttpClient,
)
from ringr_agents.idempotency import Reservation, action_scope, data_fingerprint
from ringr_agents.results import ActionOutcome
from tests.helpers import TEST_TOKEN, build_agent

URL = "https://api.ringr.debt/v1/commitment"
COMPLETE: dict[str, JsonScalar] = {"commitment_date": "2026-11-20", "committed_amount": 150.5}


def test_builds_the_exact_request_when_date_and_amount_are_known() -> None:
    harness = build_agent(DebtAgent, [COMPLETE], conversation=Conversation("conv-1"))

    result = harness.agent.handle_turn("Pagaré 150,50 € el 20 de noviembre")

    assert result.outcome is ActionOutcome.EXECUTED
    assert result.data == DebtCommitment("2026-11-20", 150.5)
    request = result.request
    assert request is not None
    assert harness.http_client.sent_requests == (request,)
    assert request.method == "POST"
    assert request.url == URL
    assert request.body == COMPLETE
    assert request.headers == {
        "commitment_date": "2026-11-20",
        "committed_amount": "150.5",
        "Content-Type": "application/json",
        "Idempotency-Key": Reservation(
            action_scope("conv-1", "debt.commitment"), data_fingerprint(COMPLETE), revision=1
        ).idempotency_key,
        "Authorization": f"Bearer {TEST_TOKEN}",
    }
    assert result.response == HttpResponse(200, {"status": "ok"})


def test_full_conversation_acts_only_when_ready_and_never_duplicates() -> None:
    harness = build_agent(
        DebtAgent,
        [
            {"commitment_date": None, "committed_amount": None},
            {"commitment_date": "2026-11-20", "committed_amount": None},
            COMPLETE,
            COMPLETE,  # el parser vuelve a ver lo mismo en el historial
            COMPLETE,
        ],
    )
    agent = harness.agent

    outcomes = [agent.handle_turn(f"mensaje {turn}").outcome for turn in range(1, 6)]

    assert outcomes == [
        ActionOutcome.NOT_READY,
        ActionOutcome.NOT_READY,
        ActionOutcome.EXECUTED,
        ActionOutcome.DUPLICATE,
        ActionOutcome.DUPLICATE,
    ]
    assert len(harness.http_client.sent_requests) == 1


def test_only_amount_is_not_enough() -> None:
    harness = build_agent(DebtAgent, [{"commitment_date": None, "committed_amount": 100}])

    result = harness.agent.handle_turn("Puedo pagar 100 €")

    assert result.outcome is ActionOutcome.NOT_READY
    assert result.data == DebtCommitment(None, 100.0)
    assert harness.http_client.sent_requests == ()


def test_missing_keys_are_treated_as_none() -> None:
    harness = build_agent(DebtAgent, [{}])

    assert harness.agent.handle_turn("hola").outcome is ActionOutcome.NOT_READY


def test_a_changed_commitment_is_registered_again() -> None:
    harness = build_agent(
        DebtAgent,
        [COMPLETE, {"commitment_date": "2026-11-20", "committed_amount": 200}],
    )

    first = harness.agent.handle_turn("150,50 el día 20")
    second = harness.agent.handle_turn("Mejor 200 €")

    assert (first.outcome, second.outcome) == (ActionOutcome.EXECUTED, ActionOutcome.EXECUTED)
    assert [r.body["committed_amount"] for r in harness.http_client.sent_requests] == [150.5, 200.0]


def test_going_back_to_an_earlier_amount_is_sent_again() -> None:
    other = {"commitment_date": "2026-11-20", "committed_amount": 200}
    harness = build_agent(DebtAgent, [COMPLETE, other, COMPLETE, COMPLETE])

    outcomes = [harness.agent.handle_turn("...").outcome for _ in range(4)]

    # La API debe quedarse con el último compromiso: 150,50 €, no 200 €.
    assert outcomes == [ActionOutcome.EXECUTED] * 3 + [ActionOutcome.DUPLICATE]
    sent = harness.http_client.sent_requests
    assert [r.body["committed_amount"] for r in sent] == [150.5, 200.0, 150.5]
    # Cada envío lleva su propia clave, aunque el primero y el tercero lleven
    # los mismos datos: si no, el servidor tomaría el tercero por un reintento.
    assert len({r.headers["Idempotency-Key"] for r in sent}) == 3


def test_amount_is_normalized_before_sending() -> None:
    harness = build_agent(
        DebtAgent, [{"commitment_date": "2026-11-20", "committed_amount": "150,5"}]
    )

    result = harness.agent.handle_turn("ciento cincuenta con cincuenta")

    assert result.outcome is ActionOutcome.EXECUTED
    assert result.request is not None
    assert result.request.body["committed_amount"] == 150.5


def test_past_date_is_invalid_and_nothing_is_sent() -> None:
    harness = build_agent(DebtAgent, [{"commitment_date": "2026-11-01", "committed_amount": 50}])

    result = harness.agent.handle_turn("Pagué el día 1")

    assert result.outcome is ActionOutcome.INVALID_DATA
    assert result.error is not None
    assert "commitment_date" in result.error
    assert result.reply == "Respuesta 1"  # el usuario recibe respuesta igualmente
    assert harness.http_client.sent_requests == ()


def test_server_error_is_retried_on_the_next_turn() -> None:
    statuses: Iterator[int] = iter([500, 200])
    client = SimulatedHttpClient(lambda request: HttpResponse(next(statuses)))
    harness = build_agent(DebtAgent, [COMPLETE, COMPLETE, COMPLETE], http_client=client)

    failed = harness.agent.handle_turn("150,50 el 20")
    retried = harness.agent.handle_turn("¿Ha quedado registrado?")
    after = harness.agent.handle_turn("Gracias")

    assert failed.outcome is ActionOutcome.FAILED
    assert failed.error == "El servidor respondió 500"
    assert retried.outcome is ActionOutcome.EXECUTED
    assert after.outcome is ActionOutcome.DUPLICATE
    # El reintento lleva la misma Idempotency-Key: el servidor puede deduplicar.
    keys = [r.headers["Idempotency-Key"] for r in client.sent_requests]
    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_network_error_is_reported_and_retried() -> None:
    calls: list[HttpRequest] = []

    def flaky(request: HttpRequest) -> HttpResponse:
        calls.append(request)
        if len(calls) == 1:
            raise HttpTransportError("timeout")
        return HttpResponse(200)

    harness = build_agent(DebtAgent, [COMPLETE, COMPLETE], http_client=SimulatedHttpClient(flaky))

    failed = harness.agent.handle_turn("150,50 el 20")
    retried = harness.agent.handle_turn("¿Sigue ahí?")

    assert failed.outcome is ActionOutcome.FAILED
    assert failed.error == "Error de red: timeout"
    assert failed.response is None
    assert retried.outcome is ActionOutcome.EXECUTED
