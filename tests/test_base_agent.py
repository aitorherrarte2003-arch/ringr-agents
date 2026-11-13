"""Comportamiento común a todos los agentes y prueba de extensibilidad."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from ringr_agents.actions import HttpAction
from ringr_agents.agents import BaseAgent, DebtAgent
from ringr_agents.conversation import Conversation, Message, Role
from ringr_agents.http_client import HttpResponse, JsonScalar, SimulatedHttpClient
from ringr_agents.idempotency import InMemoryActionLedger
from ringr_agents.models import ConversationModel, ParserModel
from ringr_agents.results import ActionOutcome
from ringr_agents.validation import optional_iso_date
from tests.helpers import build_agent

# --- Un agente nuevo en ~20 líneas: demuestra que la arquitectura escala ------


@dataclass(frozen=True)
class CallbackData:
    callback_date: str | None

    def to_payload(self) -> Mapping[str, JsonScalar]:
        return {"callback_date": self.callback_date}


class CallbackAgent(BaseAgent[CallbackData]):
    """Agente hipotético: agenda una rellamada cuando el cliente da una fecha."""

    action = HttpAction(name="callback.schedule", url="https://api.example.test/v1/callback")

    def validate_data(self, raw: Mapping[str, Any]) -> CallbackData:
        return CallbackData(
            optional_iso_date(
                raw.get("callback_date"), field="callback_date", not_before=self._today()
            )
        )

    def should_act(self, data: CallbackData) -> bool:
        return data.callback_date is not None


def test_a_new_agent_gets_the_whole_lifecycle_for_free() -> None:
    harness = build_agent(
        CallbackAgent, [{"callback_date": None}, {"callback_date": "2026-12-01"}] * 2
    )

    outcomes = [harness.agent.handle_turn("...").outcome for _ in range(4)]

    assert outcomes == [
        ActionOutcome.NOT_READY,
        ActionOutcome.EXECUTED,
        ActionOutcome.NOT_READY,
        ActionOutcome.DUPLICATE,
    ]
    assert harness.http_client.sent_requests[0].url == "https://api.example.test/v1/callback"


# --- Garantías comunes ---------------------------------------------------------


def test_agent_without_action_cannot_be_created() -> None:
    class IncompleteAgent(BaseAgent[CallbackData]):
        def validate_data(self, raw: Mapping[str, Any]) -> CallbackData:
            return CallbackData(None)

        def should_act(self, data: CallbackData) -> bool:
            return False

    with pytest.raises(TypeError, match="abstract"):
        IncompleteAgent(ConversationModel("Hola"), _ListParser())  # type: ignore[abstract]


class _ListParser:
    """Parser que incumple el contrato devolviendo una lista."""

    def parse_data(self) -> Any:
        return ["no", "soy", "un", "dict"]


def test_parser_returning_something_else_than_a_dict_is_handled() -> None:
    client = SimulatedHttpClient()
    agent = DebtAgent(ConversationModel("Hola"), _ListParser(), http_client=client)

    result = agent.handle_turn("hola")

    assert result.outcome is ActionOutcome.INVALID_DATA
    assert result.error == "parser_output: se esperaba un dict, llegó list"
    assert result.reply == "Hola"
    assert client.sent_requests == ()


def test_same_data_in_different_conversations_is_not_a_duplicate() -> None:
    shared_ledger = InMemoryActionLedger()
    shared_client = SimulatedHttpClient()
    data = [{"commitment_date": "2026-11-20", "committed_amount": 100}]

    results = [
        build_agent(
            DebtAgent,
            data,
            http_client=shared_client,
            ledger=shared_ledger,
            conversation=Conversation(conversation_id),
        ).agent.handle_turn("100 € el 20")
        for conversation_id in ("cliente-a", "cliente-b")
    ]

    assert [r.outcome for r in results] == [ActionOutcome.EXECUTED, ActionOutcome.EXECUTED]
    assert len(shared_client.sent_requests) == 2


def test_turn_result_flag() -> None:
    harness = build_agent(DebtAgent, [{"commitment_date": "2026-11-20", "committed_amount": 1}])
    assert harness.agent.handle_turn("...").action_executed is True


def test_agent_exposes_its_conversation() -> None:
    conversation = Conversation("conv-x")
    harness = build_agent(DebtAgent, [{}], conversation=conversation)

    harness.agent.handle_turn("hola")

    assert harness.agent.conversation is conversation
    assert conversation.messages == (
        Message(Role.USER, "hola"),
        Message(Role.ASSISTANT, "Respuesta 1"),
    )


class _ConversationReader:
    """Modelo que, como los reales, lleva la conversación en su estado.

    Apunta lo que ve en cada llamada, para comprobar qué hay en ella."""

    def __init__(self, conversation: Conversation) -> None:
        self._conversation = conversation
        self.seen: list[tuple[Message, ...]] = []

    def answer_user(self) -> str:
        self.seen.append(self._conversation.messages)
        return f"Respuesta {len(self.seen)}"

    def parse_data(self) -> dict[str, Any]:
        self.seen.append(self._conversation.messages)
        return {}


def test_models_read_the_conversation_from_their_state() -> None:
    conversation = Conversation()
    talker, parser = _ConversationReader(conversation), _ConversationReader(conversation)
    agent = DebtAgent(talker, parser, conversation=conversation)

    agent.handle_turn("Hola")
    agent.handle_turn("Quiero pagar")

    user_1, bot_1 = Message(Role.USER, "Hola"), Message(Role.ASSISTANT, "Respuesta 1")
    user_2, bot_2 = Message(Role.USER, "Quiero pagar"), Message(Role.ASSISTANT, "Respuesta 2")
    # El modelo conversacional ve el historial hasta el mensaje del usuario...
    assert talker.seen == [(user_1,), (user_1, bot_1, user_2)]
    # ...y el parser lo ve completo, incluida la respuesta que se acaba de dar.
    assert parser.seen == [(user_1, bot_1), (user_1, bot_1, user_2, bot_2)]


def test_a_turn_without_user_message_only_adds_the_reply() -> None:
    conversation = Conversation()
    agent = DebtAgent(
        ConversationModel("Hola, le llamo de Ringr."),
        ParserModel({"commitment_date": None, "committed_amount": None}),
        conversation=conversation,
    )

    result = agent.handle_turn()  # el agente habla primero

    assert result.outcome is ActionOutcome.NOT_READY
    assert conversation.messages == (Message(Role.ASSISTANT, "Hola, le llamo de Ringr."),)


def test_logs_count_the_turns(caplog: pytest.LogCaptureFixture) -> None:
    harness = build_agent(DebtAgent, [{}, {}])

    with caplog.at_level(logging.INFO, logger="ringr_agents"):
        harness.agent.handle_turn()
        harness.agent.handle_turn("Hola")

    assert "turn=1 " in caplog.text
    assert "turn=2 " in caplog.text


def test_logs_contain_no_personal_data(caplog: pytest.LogCaptureFixture) -> None:
    harness = build_agent(
        DebtAgent,
        [{"commitment_date": "2026-11-20", "committed_amount": 987.65}],
        conversation=Conversation("conv-log"),
    )

    with caplog.at_level(logging.INFO, logger="ringr_agents"):
        harness.agent.handle_turn("Pago 987,65 € el 20")

    assert "conv-log" in caplog.text
    assert "outcome=executed" in caplog.text
    assert "987" not in caplog.text
    assert "2026-11-20" not in caplog.text


@pytest.mark.parametrize(
    ("raw", "field", "value"),
    [
        (
            {"commitment_date": "2026-11-01", "committed_amount": 150},
            "commitment_date",
            "2026-11-01",
        ),
        (
            {"commitment_date": "2026-11-20", "committed_amount": "1.234,56"},
            "committed_amount",
            "1.234",
        ),
    ],
)
def test_rejected_data_is_never_logged(
    raw: dict[str, object], field: str, value: str, caplog: pytest.LogCaptureFixture
) -> None:
    harness = build_agent(DebtAgent, [raw])

    with caplog.at_level(logging.INFO, logger="ringr_agents"):
        result = harness.agent.handle_turn("...")

    assert result.error is not None
    assert value in result.error  # el resultado sí lo dice, para depurar
    assert f"invalid_data field={field}" in caplog.text  # el log, solo el campo
    assert value not in caplog.text


def test_a_failed_send_logs_only_the_status(caplog: pytest.LogCaptureFixture) -> None:
    harness = build_agent(
        DebtAgent,
        [{"commitment_date": "2026-11-20", "committed_amount": 987.65}],
        http_client=SimulatedHttpClient(lambda request: HttpResponse(503)),
    )

    with caplog.at_level(logging.INFO, logger="ringr_agents"):
        harness.agent.handle_turn("...")

    assert "cause=http_status=503" in caplog.text
    assert "987" not in caplog.text
