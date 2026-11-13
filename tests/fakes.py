"""Cliente de Anthropic falso: imita lo justo del SDK para testear sin red.

Devuelve respuestas preparadas en orden y guarda los argumentos de cada
llamada para comprobar qué se le envió a la API.
"""

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import anthropic


@dataclass
class FakeBlock:
    type: str
    text: str = ""


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"


def text_response(text: str, *, stop_reason: str = "end_turn") -> FakeResponse:
    # Un bloque de razonamiento delante, como devuelve el modelo real: el
    # adaptador debe ignorarlo y quedarse solo con el texto.
    return FakeResponse([FakeBlock("thinking"), FakeBlock("text", text)], stop_reason)


@dataclass
class FakeMessages:
    # Cada llamada consume una: si es una excepción, se lanza (simula un fallo de la API).
    responses: deque[FakeResponse | Exception]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class FakeAnthropic:
    def __init__(
        self, *responses: FakeResponse | Exception, api_key: str | None = "fake-key"
    ) -> None:
        self.messages = FakeMessages(deque(responses))
        self.beta = SimpleNamespace(messages=self.messages)
        # Credenciales, como las tiene el cliente real (ver cli._has_credentials).
        self.api_key = api_key
        self.auth_token: str | None = None
        self.credentials: object | None = None

    def as_client(self) -> anthropic.Anthropic:
        """El mismo objeto, presentado con el tipo que esperan los adaptadores."""
        return cast(anthropic.Anthropic, self)
