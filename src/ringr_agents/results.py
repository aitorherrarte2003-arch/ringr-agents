"""Resultado de un turno.

`handle_turn` no imprime nada ni devuelve solo un texto: devuelve un objeto
con todo lo que ha pasado en el turno. Así es fácil de testear, de loguear y
de depurar ("¿por qué no se registró el compromiso?" → mira `outcome` y `error`).
"""

from dataclasses import dataclass
from enum import StrEnum

from ringr_agents.http_client import HttpRequest, HttpResponse


class ActionOutcome(StrEnum):
    """Qué pasó con la acción externa en este turno."""

    EXECUTED = "executed"
    """Se envió la petición y el servidor respondió 2xx."""

    NOT_READY = "not_ready"
    """Los datos son válidos pero aún faltan campos para actuar."""

    DUPLICATE = "duplicate"
    """Esos datos ya se enviaron, o hay otro envío en curso en esta conversación."""

    INVALID_DATA = "invalid_data"
    """El ParserModel devolvió datos incorrectos; no se actúa."""

    FAILED = "failed"
    """No se pudo enviar (error HTTP, de red o del registro anti-duplicados). Se
    reintentará en el siguiente turno, si lo hay y los datos siguen pendientes."""


@dataclass(frozen=True, slots=True)
class TurnResult[DataT]:
    """Todo lo que ocurrió en un turno."""

    reply: str
    outcome: ActionOutcome
    data: DataT | None = None
    request: HttpRequest | None = None
    response: HttpResponse | None = None
    error: str | None = None

    @property
    def action_executed(self) -> bool:
        return self.outcome is ActionOutcome.EXECUTED
