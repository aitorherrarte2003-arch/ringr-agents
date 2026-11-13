"""Tipos HTTP y cliente simulado.

Los agentes no saben nada de red: construyen un `HttpRequest` y se lo pasan a
un `HttpClient`. El enunciado pide no hacer llamadas reales, así que usamos
`SimulatedHttpClient`, que guarda las peticiones y responde 200 OK. Un cliente
real (por ejemplo con httpx) cumpliría el mismo contrato y bastaría con
inyectarlo en el agente.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

# Valores que admitimos en el body JSON de las peticiones.
type JsonScalar = str | int | float | bool | None

_SENSITIVE_HEADERS = frozenset({"authorization"})


@dataclass(frozen=True, slots=True, repr=False)
class HttpRequest:
    """Petición HTTP ya construida, lista para enviar. Es inmutable."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        # `frozen` impide reasignar atributos, pero un dict seguiría siendo
        # modificable por dentro. Lo copiamos y lo envolvemos en una vista de
        # solo lectura para que nadie pueda cambiar la petición después.
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        object.__setattr__(self, "body", MappingProxyType(dict(self.body)))

    def json_body(self) -> str:
        """Body serializado tal y como viajaría por la red."""
        return json.dumps(dict(self.body), ensure_ascii=False)

    def redacted_headers(self) -> dict[str, str]:
        """Cabeceras con el token oculto, para mostrarlas o loguearlas."""
        return {
            name: "***" if name.lower() in _SENSITIVE_HEADERS else value
            for name, value in self.headers.items()
        }

    def __repr__(self) -> str:
        # Ocultamos el token para que no acabe en logs al imprimir la petición.
        return (
            f"HttpRequest(method={self.method!r}, url={self.url!r}, "
            f"headers={self.redacted_headers()!r}, body={dict(self.body)!r})"
        )


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Respuesta HTTP mínima: código de estado y body opcional."""

    status_code: int
    body: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True si el código es 2xx."""
        return 200 <= self.status_code < 300


class HttpTransportError(Exception):
    """La petición no obtuvo respuesta (timeout, DNS, conexión rechazada...)."""


class HttpClient(Protocol):
    """Contrato de cualquier cliente HTTP que use un agente."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Envía la petición. Lanza `HttpTransportError` si no hay respuesta."""
        ...


def _always_ok(request: HttpRequest) -> HttpResponse:
    return HttpResponse(status_code=200, body={"status": "ok"})


class SimulatedHttpClient:
    """Cliente que no sale a la red.

    Guarda cada petición recibida y devuelve 200 OK. En los tests se le puede
    pasar un `responder` para simular errores (500, timeouts...).
    """

    def __init__(self, responder: Callable[[HttpRequest], HttpResponse] | None = None) -> None:
        self._responder = responder or _always_ok
        self._sent: list[HttpRequest] = []

    @property
    def sent_requests(self) -> tuple[HttpRequest, ...]:
        """Peticiones que han intentado enviarse, en orden."""
        return tuple(self._sent)

    def send(self, request: HttpRequest) -> HttpResponse:
        self._sent.append(request)
        return self._responder(request)
