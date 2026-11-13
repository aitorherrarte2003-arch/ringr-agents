from urllib.parse import unquote

from ringr_agents.agents import AssistanceAgent, AssistanceRequest
from ringr_agents.conversation import Conversation
from ringr_agents.http_client import HttpResponse, SimulatedHttpClient
from ringr_agents.idempotency import Reservation, action_scope, data_fingerprint
from ringr_agents.results import ActionOutcome
from tests.helpers import TEST_TOKEN, build_agent

URL = "https://api.ringr.assistance/v1/request"


def test_builds_the_exact_request_when_there_is_a_request() -> None:
    harness = build_agent(
        AssistanceAgent, [{"request": "Cambiar mi tarifa"}], conversation=Conversation("conv-9")
    )

    result = harness.agent.handle_turn("Quiero cambiar mi tarifa")

    assert result.outcome is ActionOutcome.EXECUTED
    request = result.request
    assert request is not None
    assert request.method == "POST"
    assert request.url == URL
    assert request.body == {"request": "Cambiar mi tarifa"}
    assert request.headers == {
        "request": "Cambiar mi tarifa",  # mismo nombre y mismo valor que el dato
        "Content-Type": "application/json",
        "Idempotency-Key": Reservation(
            action_scope("conv-9", "assistance.request"), data_fingerprint({}), revision=1
        ).idempotency_key,
        "Authorization": f"Bearer {TEST_TOKEN}",
    }


def test_questions_without_a_request_do_not_trigger_actions() -> None:
    harness = build_agent(AssistanceAgent, [{"request": None}, {"request": "   "}, {}])

    outcomes = [harness.agent.handle_turn("¿Qué horario tenéis?").outcome for _ in range(3)]

    assert outcomes == [ActionOutcome.NOT_READY] * 3
    assert harness.http_client.sent_requests == ()


def test_one_request_per_conversation_even_if_the_llm_rewords_it() -> None:
    harness = build_agent(
        AssistanceAgent,
        [
            {"request": "Cambio de tarifa a fibra de 1 Gb"},
            {"request": "Cambio de tarifa a fibra 1 Gb"},  # el LLM lo redacta distinto
            {"request": "cambiar la tarifa a la fibra de 1 Gb"},
        ],
    )

    outcomes = [harness.agent.handle_turn("...").outcome for _ in range(3)]

    assert outcomes == [ActionOutcome.EXECUTED, ActionOutcome.DUPLICATE, ActionOutcome.DUPLICATE]
    assert [r.body["request"] for r in harness.http_client.sent_requests] == [
        "Cambio de tarifa a fibra de 1 Gb"
    ]


def test_a_failed_request_is_retried_even_if_reworded() -> None:
    statuses = iter([503, 200])
    harness = build_agent(
        AssistanceAgent,
        [{"request": "Cambio a fibra 1 Gb"}, {"request": "Cambiar a la fibra de 1 Gb"}],
        http_client=SimulatedHttpClient(lambda request: HttpResponse(next(statuses))),
    )

    failed = harness.agent.handle_turn("...")
    retried = harness.agent.handle_turn("...")

    assert (failed.outcome, retried.outcome) == (ActionOutcome.FAILED, ActionOutcome.EXECUTED)
    first, retry = harness.http_client.sent_requests
    # Es el mismo envío: misma clave y exactamente el mismo contenido, aunque el
    # LLM lo haya redactado distinto (un servidor rechazaría la clave con otro body).
    assert retry.headers["Idempotency-Key"] == first.headers["Idempotency-Key"]
    assert retry.body == first.body == {"request": "Cambio a fibra 1 Gb"}


def test_text_with_accents_and_line_breaks_is_safe_in_headers() -> None:
    text = "Revisar mi factura\nde información"
    harness = build_agent(AssistanceAgent, [{"request": text}])

    result = harness.agent.handle_turn("...")

    assert result.request is not None
    header = result.request.headers["request"]
    assert header.isascii()
    assert "\n" not in header
    assert unquote(header) == text
    assert result.request.body["request"] == text  # el body lleva el texto original


def test_non_text_request_is_invalid() -> None:
    harness = build_agent(AssistanceAgent, [{"request": 123}])

    result = harness.agent.handle_turn("...")

    assert result.outcome is ActionOutcome.INVALID_DATA
    assert result.data is None
    assert harness.http_client.sent_requests == ()


def test_data_is_exposed_in_the_result() -> None:
    harness = build_agent(AssistanceAgent, [{"request": None}])

    assert harness.agent.handle_turn("hola").data == AssistanceRequest(None)
