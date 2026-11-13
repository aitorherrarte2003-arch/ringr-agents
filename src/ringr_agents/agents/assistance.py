"""Agente de atención al cliente: registra solicitudes para agentes humanos."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from ringr_agents.actions import HttpAction
from ringr_agents.agents.base import BaseAgent
from ringr_agents.http_client import JsonScalar
from ringr_agents.validation import optional_text


@dataclass(frozen=True, slots=True)
class AssistanceRequest:
    """Solicitud ya validada. None si el usuario aún no ha pedido nada."""

    request: str | None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> Self:
        """Valida y normaliza la salida del ParserModel."""
        return cls(request=optional_text(raw.get("request"), field="request"))

    def to_payload(self) -> Mapping[str, JsonScalar]:
        return {"request": self.request}


class AssistanceAgent(BaseAgent[AssistanceRequest]):
    """Registra la solicitud en cuanto el usuario plantea una."""

    action = HttpAction(name="assistance.request", url="https://api.ringr.assistance/v1/request")

    def validate_data(self, raw: Mapping[str, Any]) -> AssistanceRequest:
        return AssistanceRequest.from_raw(raw)

    def should_act(self, data: AssistanceRequest) -> bool:
        return data.request is not None

    def dedup_payload(self, data: AssistanceRequest) -> Mapping[str, JsonScalar]:
        # Una solicitud por conversación. El texto lo redacta el LLM y cambia de
        # un turno a otro ("Cambio a fibra 1 Gb" / "Cambiar a la fibra de 1 Gb"):
        # si contara para los duplicados, una misma petición se enviaría varias
        # veces. El agente humano que la atiende lee la conversación entera.
        return {}
