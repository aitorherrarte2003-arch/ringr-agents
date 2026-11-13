"""Acciones externas: a qué endpoint se llama y cómo se construye la petición.

Una `HttpAction` describe una integración (nombre, URL, método) y sabe
construir el `HttpRequest` a partir de los datos validados. Añadir una
integración nueva es crear otra `HttpAction`; no hay que tocar los agentes
existentes.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote

from ringr_agents.auth import Authenticator
from ringr_agents.http_client import HttpRequest, JsonScalar

# Cabeceras que pone el sistema. Un dato de la conversación nunca puede usar
# uno de estos nombres (comparamos en minúsculas: en HTTP no distinguen mayúsculas).
_RESERVED_HEADERS = frozenset({"authorization", "content-type", "idempotency-key"})

# Lo que puede ir tal cual en una cabecera: ASCII visible y el espacio (RFC 9110).
# Menos el "%", que es el que marca un carácter codificado.
_HEADER_SAFE = "".join(chr(code) for code in range(0x20, 0x7F) if chr(code) != "%")
# Espacios al principio o al final del valor.
_EDGE_SPACES = re.compile(r"^ +| +$")


@dataclass(frozen=True, slots=True)
class HttpAction:
    """Integración HTTP que un agente puede ejecutar."""

    name: str
    url: str
    method: str = "POST"

    def build_request(
        self,
        payload: Mapping[str, JsonScalar],
        *,
        auth: Authenticator,
        idempotency_key: str,
    ) -> HttpRequest:
        """Construye la petición completa: URL, cabeceras y body.

        Cabeceras, en este orden:
        1. Los datos del ParserModel, con los mismos nombres (lo pide el enunciado).
        2. Content-Type e Idempotency-Key.
        3. Authorization, al final: nada puede sobrescribir la autenticación.
        """
        headers = payload_headers(payload)
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = idempotency_key
        headers.update(auth.auth_headers())
        return HttpRequest(method=self.method, url=self.url, headers=headers, body=payload)


def payload_headers(payload: Mapping[str, JsonScalar]) -> dict[str, str]:
    """Convierte los datos en cabeceras HTTP con el mismo nombre.

    Los valores None no se envían como cabecera (no hay nada que comunicar).
    """
    headers: dict[str, str] = {}
    for name, value in payload.items():
        if name.lower() in _RESERVED_HEADERS:
            raise ValueError(f"El campo {name!r} coincide con una cabecera reservada.")
        if value is not None:
            headers[name] = encode_header_value(value)
    return headers


def encode_header_value(value: JsonScalar) -> str:
    """Convierte un valor en texto apto para una cabecera HTTP.

    Las cabeceras HTTP solo admiten ASCII visible y espacios. Un texto del
    usuario puede traer tildes ("información") o saltos de línea, que además
    permitirían inyectar cabeceras falsas. Solo esos caracteres (y el propio
    "%") se codifican en formato URL (percent-encoding, RFC 3986); el resto va
    tal cual. "Quiero darme de baja", "2026-11-20" y "150.5" no cambian, e
    "información" pasa a "informaci%C3%B3n". El receptor recupera siempre el
    original con un simple `unquote`. El body lleva siempre el valor original.
    """
    text = value if isinstance(value, str) else json.dumps(value)
    encoded = quote(text, safe=_HEADER_SAFE)
    # HTTP recorta los espacios de los extremos: esos sí se codifican, o se perderían.
    return _EDGE_SPACES.sub(lambda match: "%20" * len(match.group()), encoded)
