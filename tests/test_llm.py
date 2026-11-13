"""Adaptadores de Claude y perfiles de agente (con un cliente falso, sin red)."""

import dataclasses
import json
from datetime import date

import anthropic
import httpx2  # la versión de httpx que usa por dentro el SDK de Anthropic
import pytest

from ringr_agents.agents import AssistanceRequest, DebtAgent, DebtCommitment
from ringr_agents.conversation import Conversation, Message, Role
from ringr_agents.llm.claude import DEFAULT_MODEL, ClaudeConversationModel, ClaudeParserModel
from ringr_agents.llm.profiles import ASSISTANCE_PROFILE, DEBT_PROFILE, AgentProfile
from ringr_agents.models import ModelResponseError, ModelUnavailableError
from ringr_agents.results import ActionOutcome
from tests.fakes import FakeAnthropic, FakeBlock, FakeResponse, text_response

HISTORY = (
    Message(Role.USER, "Hola"),
    Message(Role.ASSISTANT, "¿Cuándo podría pagar?"),
    Message(Role.USER, "El viernes, 300 €"),
)
SCHEMA = DEBT_PROFILE.parser_schema
START = ("user", "(Empieza la llamada.)")
SILENT = ("user", "(El cliente no dice nada.)")


def talker(
    fake: FakeAnthropic, conversation: Conversation | None = None
) -> ClaudeConversationModel:
    """ClaudeConversationModel con el historial de ejemplo (u otra conversación)."""
    return ClaudeConversationModel(
        fake.as_client(),
        conversation=conversation if conversation is not None else Conversation(messages=HISTORY),
        instructions="Eres un agente.",
    )


def parser(fake: FakeAnthropic, conversation: Conversation | None = None) -> ClaudeParserModel:
    """ClaudeParserModel con el historial de ejemplo (u otra conversación)."""
    return ClaudeParserModel(
        fake.as_client(),
        conversation=conversation if conversation is not None else Conversation(messages=HISTORY),
        instructions="Extrae datos.",
        schema=SCHEMA,
        today=lambda: date(2026, 11, 10),
    )


class TestConversationModel:
    def test_sends_history_and_returns_the_text(self) -> None:
        fake = FakeAnthropic(text_response("  Perfecto, el viernes.  "))

        reply = talker(fake).answer_user()

        assert reply == "Perfecto, el viernes."
        call = fake.messages.calls[0]
        assert call["model"] == DEFAULT_MODEL
        assert call["system"] == "Eres un agente."
        assert call["messages"] == [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "¿Cuándo podría pagar?"},
            {"role": "user", "content": "El viernes, 300 €"},
        ]
        assert call["output_config"] == {"effort": "low"}
        assert call["fallbacks"] == "default"
        assert call["betas"] == ["server-side-fallback-2026-07-01"]

    def test_reads_the_conversation_as_it_grows(self) -> None:
        """La conversación es parte de su estado: ve lo que se añade después de crearlo."""
        fake = FakeAnthropic(text_response("uno"), text_response("dos"))
        conversation = Conversation()
        model = talker(fake, conversation)

        conversation.add_user_message("Hola")
        model.answer_user()
        conversation.add_user_message("¿Sigue ahí?")
        model.answer_user()

        assert fake.messages.calls[1]["messages"] == [
            {"role": "user", "content": "Hola"},
            {"role": "user", "content": "¿Sigue ahí?"},
        ]

    @pytest.mark.parametrize(
        ("history", "expected"),
        [
            ((), [START]),
            ((Message(Role.ASSISTANT, "Hola"),), [START, ("assistant", "Hola"), SILENT]),
            (
                (Message(Role.USER, "¿Sí?"), Message(Role.ASSISTANT, "Hola")),
                [("user", "¿Sí?"), ("assistant", "Hola"), SILENT],
            ),
        ],
        ids=["vacía", "empieza el agente", "el agente habla dos veces"],
    )
    def test_the_call_always_starts_and_ends_with_the_user(
        self, history: tuple[Message, ...], expected: list[tuple[str, str]]
    ) -> None:
        """La API espera que la conversación empiece y termine con el usuario: si no,
        se añade una nota (turnos sin mensaje, o llamadas que empieza el agente)."""
        fake = FakeAnthropic(text_response("Hola"))

        talker(fake, Conversation(messages=history)).answer_user()

        sent = fake.messages.calls[0]["messages"]
        assert [(message["role"], message["content"]) for message in sent] == expected

    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (text_response("x", stop_reason="refusal"), "rechazó"),
            (text_response("x", stop_reason="max_tokens"), "límite de tokens"),
            (text_response("x", stop_reason="model_context_window_exceeded"), "límite de tokens"),
            (FakeResponse([FakeBlock("thinking")]), "no devolvió texto"),
            (text_response("   "), "no devolvió texto"),
        ],
    )
    def test_unusable_responses_raise(self, response: FakeResponse, message: str) -> None:
        with pytest.raises(ModelResponseError, match=message):
            talker(FakeAnthropic(response)).answer_user()


class TestParserModel:
    def test_sends_transcript_schema_and_today(self) -> None:
        output = {"commitment_date": "2026-11-13", "committed_amount": 300}
        fake = FakeAnthropic(text_response(json.dumps(output)))

        assert parser(fake).parse_data() == output

        call = fake.messages.calls[0]
        assert call["system"] == "Extrae datos.\n\nFecha de hoy: 2026-11-10."
        assert call["messages"] == [
            {
                "role": "user",
                "content": "Conversación:\n\nCliente: Hola\n"
                "Agente: ¿Cuándo podría pagar?\nCliente: El viernes, 300 €",
            }
        ]
        assert call["output_config"] == {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": SCHEMA},
        }

    @pytest.mark.parametrize(
        ("text", "message"),
        [("{no es json", "JSON válido"), ("[1, 2]", "objeto JSON")],
    )
    def test_bad_json_raises(self, text: str, message: str) -> None:
        with pytest.raises(ModelResponseError, match=message):
            parser(FakeAnthropic(text_response(text))).parse_data()


def claude_agent(fake: FakeAnthropic) -> DebtAgent:
    """DebtAgent con los dos modelos de Claude compartiendo su conversación."""
    conversation = Conversation()
    return DebtAgent(
        talker(fake, conversation),
        parser(fake, conversation),
        conversation=conversation,
        today=lambda: date(2026, 11, 10),
    )


def test_agent_works_end_to_end_with_the_claude_adapters() -> None:
    """Los agentes no cambian: solo se inyectan otros modelos."""
    fake = FakeAnthropic(
        text_response("Perfecto, 300 € el viernes 13."),
        text_response('{"commitment_date": "2026-11-13", "committed_amount": 300}'),
    )

    result = claude_agent(fake).handle_turn("El viernes le pago 300")

    assert result.outcome is ActionOutcome.EXECUTED
    assert result.data == DebtCommitment("2026-11-13", 300.0)
    assert result.request is not None
    # El parser leyó la conversación entera, incluida la respuesta recién dada.
    transcript = fake.messages.calls[1]["messages"][0]["content"]
    assert transcript.endswith(
        "Cliente: El viernes le pago 300\nAgente: Perfecto, 300 € el viernes 13."
    )


def api_failure(error_class: type[anthropic.APIStatusError], status: int) -> Exception:
    """Error de la API tal y como lo lanza el SDK."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return error_class("fallo de la API", response=response, body=None)


TRANSIENT_ERRORS = [
    anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
    api_failure(anthropic.RateLimitError, 429),
    api_failure(anthropic.InternalServerError, 529),
]


@pytest.mark.parametrize("error", TRANSIENT_ERRORS)
def test_transient_api_errors_become_model_errors(error: Exception) -> None:
    with pytest.raises(ModelUnavailableError, match="no respondió"):
        talker(FakeAnthropic(error)).answer_user()
    with pytest.raises(ModelUnavailableError, match="no respondió"):
        parser(FakeAnthropic(error)).parse_data()
    assert issubclass(ModelUnavailableError, ModelResponseError)  # el agente lo gestiona igual


def test_configuration_errors_are_not_hidden() -> None:
    error = api_failure(anthropic.AuthenticationError, 401)

    with pytest.raises(anthropic.AuthenticationError):
        talker(FakeAnthropic(error)).answer_user()


def test_an_api_outage_in_the_parser_does_not_break_the_turn() -> None:
    fake = FakeAnthropic(text_response("Le escucho."), TRANSIENT_ERRORS[0])

    result = claude_agent(fake).handle_turn("Hola")

    assert result.outcome is ActionOutcome.INVALID_DATA
    assert result.reply == "Le escucho."  # el cliente tiene su respuesta igualmente


def test_a_parser_failure_does_not_break_the_turn() -> None:
    fake = FakeAnthropic(text_response("Le escucho."), text_response("no es json"))

    result = claude_agent(fake).handle_turn("Hola")

    assert result.outcome is ActionOutcome.INVALID_DATA
    assert result.reply == "Le escucho."
    assert result.error is not None
    assert "JSON válido" in result.error


@pytest.mark.parametrize(
    ("profile", "data_class"),
    [(DEBT_PROFILE, DebtCommitment), (ASSISTANCE_PROFILE, AssistanceRequest)],
)
def test_parser_schema_matches_the_data_class(profile: AgentProfile, data_class: type) -> None:
    """El esquema que se pide al LLM y los datos que valida el agente no pueden
    desincronizarse: si alguien añade un campo a uno, este test falla."""
    fields = {field.name for field in dataclasses.fields(data_class)}
    schema = profile.parser_schema

    assert set(schema["properties"]) == fields
    assert set(schema["required"]) == fields
    assert schema["additionalProperties"] is False
    for property_schema in schema["properties"].values():
        assert {"type": "null"} in property_schema["anyOf"]  # todos admiten "aún no se sabe"
