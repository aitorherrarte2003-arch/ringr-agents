"""Cliente HTTP real con httpx (instalar con `pip install -e ".[http]"`).

Cumple el mismo contrato `HttpClient` que el simulado: en producción se
inyectaría este en los agentes y nada más cambiaría.

Reintenta los fallos transitorios (errores de red, 429 y 5xx) esperando cada
vez el doble (backoff exponencial). Cada petición lleva su `Idempotency-Key`:
si el primer intento sí llegó, un servidor que respete esa cabecera reconoce
el duplicado y no registra la acción dos veces.
"""

import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

import httpx

from ringr_agents.http_client import HttpRequest, HttpResponse, HttpTransportError


class HttpxClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        timeout_seconds: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries no puede ser negativo.")
        self._client = client if client is not None else httpx.Client(timeout=timeout_seconds)
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        # Inyectar `sleep` permite testear los reintentos sin esperar de verdad.
        self._sleep = sleep

    def send(self, request: HttpRequest) -> HttpResponse:
        delay = self._backoff_seconds
        for _ in range(self._max_retries):
            outcome = self._attempt(request)
            if not _is_retryable(outcome):
                return _resolve(outcome)
            self._sleep(delay)
            delay *= 2
        # Último intento: su resultado se devuelve (o se lanza) sea cual sea.
        return _resolve(self._attempt(request))

    def _attempt(self, request: HttpRequest) -> HttpResponse | HttpTransportError:
        """Un intento de envío. Devuelve el error en lugar de lanzarlo para
        que `send` decida si reintenta."""
        try:
            raw = self._client.request(
                request.method,
                request.url,
                headers=dict(request.headers),
                content=request.json_body().encode("utf-8"),
            )
        except httpx.RequestError as error:  # red, timeouts, respuesta corrupta...
            return HttpTransportError(f"{type(error).__name__}: {error}")
        return HttpResponse(status_code=raw.status_code, body=_json_body(raw))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _is_retryable(outcome: HttpResponse | HttpTransportError) -> bool:
    """Se reintentan los fallos que pueden arreglarse solos; un 4xx no."""
    if isinstance(outcome, HttpTransportError):
        return True
    return outcome.status_code == 429 or outcome.status_code >= 500


def _resolve(outcome: HttpResponse | HttpTransportError) -> HttpResponse:
    if isinstance(outcome, HttpTransportError):
        raise outcome
    return outcome


def _json_body(raw: httpx.Response) -> dict[str, Any]:
    """Body de la respuesta como dict; vacío si no es un objeto JSON."""
    try:
        body = raw.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
