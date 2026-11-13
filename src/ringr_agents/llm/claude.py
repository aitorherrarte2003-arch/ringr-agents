"""ConversationModel y ParserModel reales con Claude (SDK oficial de Anthropic).

Cumplen exactamente los mismos contratos que los modelos simulados, así que
los agentes no cambian: basta con inyectar estos en lugar de los otros.

Decisiones:
- Como pide el enunciado, todo lo que necesitan va en su estado interno: el
  cliente, las instrucciones, el esquema y la conversación (`Conversation`),
  la misma que el agente va rellenando. Por eso `answer_user()` y
  `parse_data()` no reciben argumentos.
- El cliente de Anthropic se inyecta desde fuera (los tests usan uno falso).
- El parser usa salida estructurada (`output_config.format`): la API garantiza
  un JSON que cumple el esquema, así que no hay que "adivinar" el formato.
- Se pasa la fecha de hoy al parser para resolver "el viernes que viene".
- `fallbacks="default"`: si el modelo rechaza una petición por sus filtros de
  seguridad, la API la reintenta con otro modelo en la misma llamada.
- Los fallos pasajeros de la API (red, saturación, 5xx), que el SDK ya ha
  reintentado por su cuenta, se convierten en `ModelUnavailableError` (un tipo
  de `ModelResponseError`): el agente
  los gestiona sin romper la conversación. Los de configuración (API key
  incorrecta, petición mal formada) no: hay que arreglarlos, no esperar.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from typing import Any, Literal

import anthropic
from anthropic.types.beta import BetaMessage, BetaMessageParam

from ringr_agents.conversation import Conversation, Message, Role
from ringr_agents.models import ModelResponseError, ModelUnavailableError

# Sonnet: buena comprensión con menos latencia y coste que Opus (en voz, la latencia importa).
DEFAULT_MODEL = "claude-sonnet-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Tope de tokens de la respuesta (incluye el razonamiento interno del modelo).
# Es un límite, no un gasto: el modelo usa solo lo que necesita.
_MAX_TOKENS = 16000
# La API espera que la conversación empiece y termine con un mensaje del
# usuario. Si el agente habla primero (una llamada de cobros la hace él) o dos
# veces seguidas (un turno sin mensaje del usuario), se añade una nota.
_CALL_STARTS = "(Empieza la llamada.)"
_USER_SILENT = "(El cliente no dice nada.)"

type Effort = Literal["low", "medium", "high"]

# Errores que pueden arreglarse solos esperando un poco.
_TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,  # sin conexión o tiempo agotado
    anthropic.RateLimitError,  # 429: demasiadas peticiones
    anthropic.InternalServerError,  # 5xx, incluido 529 (API saturada)
)


class ClaudeConversationModel:
    """Genera la respuesta del agente al usuario."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        conversation: Conversation,
        instructions: str,
        model: str = DEFAULT_MODEL,
        effort: Effort = "low",  # respuestas cortas de voz: prima la velocidad
    ) -> None:
        self._client = client
        self._conversation = conversation
        self._instructions = instructions
        self._model = model
        self._effort: Effort = effort

    def answer_user(self) -> str:
        with _transient_errors_as_model_errors():
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=self._instructions,
                messages=_api_messages(self._conversation.messages),
                output_config={"effort": self._effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        return _text_of(response)


class ClaudeParserModel:
    """Extrae los datos de la conversación como un dict que cumple `schema`."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        conversation: Conversation,
        instructions: str,
        schema: dict[str, Any],
        model: str = DEFAULT_MODEL,
        effort: Effort = "medium",  # resolver fechas relativas pide algo más de cuidado
        today: Callable[[], date] = date.today,
    ) -> None:
        self._client = client
        self._conversation = conversation
        self._instructions = instructions
        self._schema = schema
        self._model = model
        self._effort: Effort = effort
        self._today = today

    def parse_data(self) -> dict[str, Any]:
        with _transient_errors_as_model_errors():
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=f"{self._instructions}\n\nFecha de hoy: {self._today().isoformat()}.",
                messages=[{"role": "user", "content": _transcript(self._conversation.messages)}],
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": self._schema},
                },
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        try:
            data = json.loads(_text_of(response))
        except ValueError as error:  # JSONDecodeError es un tipo de ValueError
            raise ModelResponseError(f"El parser no devolvió JSON válido: {error}") from None
        if not isinstance(data, dict):
            raise ModelResponseError("El parser no devolvió un objeto JSON.")
        return data


@contextmanager
def _transient_errors_as_model_errors() -> Iterator[None]:
    """Convierte los fallos pasajeros de la API en `ModelUnavailableError`."""
    try:
        yield
    except _TRANSIENT_ERRORS as error:
        raise ModelUnavailableError(f"La API de Anthropic no respondió: {error}") from error


def _api_messages(messages: Sequence[Message]) -> list[BetaMessageParam]:
    """Historial en el formato de la API: empieza y termina con el usuario."""
    api_messages = [_to_api_message(message) for message in messages]
    if not messages or messages[0].role is not Role.USER:
        api_messages.insert(0, {"role": "user", "content": _CALL_STARTS})
    if messages and messages[-1].role is not Role.USER:
        api_messages.append({"role": "user", "content": _USER_SILENT})
    return api_messages


def _to_api_message(message: Message) -> BetaMessageParam:
    """Message propio → formato de la API de Anthropic."""
    if message.role is Role.USER:
        return {"role": "user", "content": message.content}
    return {"role": "assistant", "content": message.content}


def _transcript(messages: Sequence[Message]) -> str:
    """Conversación como texto: el parser la lee entera en un único mensaje."""
    speaker = {Role.USER: "Cliente", Role.ASSISTANT: "Agente"}
    lines = [f"{speaker[message.role]}: {message.content}" for message in messages]
    return "Conversación:\n\n" + "\n".join(lines)


def _text_of(response: BetaMessage) -> str:
    """Texto de la respuesta, comprobando antes que no se cortó ni se rechazó."""
    if response.stop_reason == "refusal":
        raise ModelResponseError("El modelo rechazó la petición.")
    if response.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise ModelResponseError("La respuesta se cortó por el límite de tokens.")
    # Si hubo fallback, sin streaming la API ya ha quitado la parte rechazada:
    # todos los bloques de texto son de la respuesta buena.
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ModelResponseError("El modelo no devolvió texto.")
    return text
