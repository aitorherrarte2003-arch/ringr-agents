"""Utilidades compartidas por los tests."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from ringr_agents.agents.base import BaseAgent
from ringr_agents.auth import BearerTokenAuth
from ringr_agents.conversation import Conversation
from ringr_agents.http_client import SimulatedHttpClient
from ringr_agents.idempotency import ActionLedger
from ringr_agents.models import ConversationModel, ParserModel

# "Hoy" fijo en todos los tests: así las fechas pasadas/futuras no dependen
# del día en que se ejecuten.
TODAY = date(2026, 11, 10)
TEST_TOKEN = "test-token-123"


@dataclass
class Harness[AgentT]:
    """Un agente junto con sus dependencias simuladas, para inspeccionarlas."""

    agent: AgentT
    http_client: SimulatedHttpClient
    conversation_model: ConversationModel
    parser_model: ParserModel


def build_agent[AgentT: BaseAgent[Any]](
    agent_class: type[AgentT],
    parser_outputs: Sequence[Mapping[str, Any]],
    *,
    http_client: SimulatedHttpClient | None = None,
    ledger: ActionLedger | None = None,
    conversation: Conversation | None = None,
) -> Harness[AgentT]:
    """Crea un agente con un guion: una salida del parser por turno."""
    client = http_client if http_client is not None else SimulatedHttpClient()
    conversation_model = ConversationModel(
        f"Respuesta {turn}" for turn in range(1, len(parser_outputs) + 1)
    )
    parser_model = ParserModel(parser_outputs)
    agent = agent_class(
        conversation_model,
        parser_model,
        http_client=client,
        auth=BearerTokenAuth(TEST_TOKEN),
        ledger=ledger,
        conversation=conversation,
        today=lambda: TODAY,
    )
    return Harness(agent, client, conversation_model, parser_model)
