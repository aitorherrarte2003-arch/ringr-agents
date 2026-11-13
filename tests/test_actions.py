from urllib.parse import unquote

import pytest

from ringr_agents.actions import HttpAction, encode_header_value, payload_headers
from ringr_agents.auth import BearerTokenAuth

ACTION = HttpAction(name="test.action", url="https://api.example.test/v1/thing")
AUTH = BearerTokenAuth("secret-token")


def test_build_request_sets_method_url_and_body() -> None:
    request = ACTION.build_request({"a": "x", "b": 1.5}, auth=AUTH, idempotency_key="key-1")

    assert request.method == "POST"
    assert request.url == "https://api.example.test/v1/thing"
    assert request.body == {"a": "x", "b": 1.5}


def test_build_request_sets_all_headers() -> None:
    request = ACTION.build_request({"a": "x", "b": 1.5}, auth=AUTH, idempotency_key="key-1")

    assert request.headers == {
        "a": "x",
        "b": "1.5",
        "Content-Type": "application/json",
        "Idempotency-Key": "key-1",
        "Authorization": "Bearer secret-token",
    }


def test_none_values_go_in_body_but_not_in_headers() -> None:
    request = ACTION.build_request({"a": None}, auth=AUTH, idempotency_key="k")

    assert request.body == {"a": None}
    assert "a" not in request.headers


@pytest.mark.parametrize(
    "name", ["Authorization", "authorization", "Content-Type", "IDEMPOTENCY-KEY"]
)
def test_payload_cannot_use_reserved_header_names(name: str) -> None:
    with pytest.raises(ValueError, match="reservada"):
        payload_headers({name: "x"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-11-20", "2026-11-20"),
        (150.5, "150.5"),
        (150, "150"),
        (True, "true"),
        # ASCII visible y espacios van tal cual: la cabecera es igual que el dato.
        ("Quiero darme de baja", "Quiero darme de baja"),
        ("información", "informaci%C3%B3n"),
        # El "%" marca lo codificado: si viene en el dato, también se codifica.
        ("100% seguro", "100%25 seguro"),
        # HTTP recorta los espacios de los extremos: se codifican para no perderlos.
        ("  hola ", "%20%20hola%20"),
        # Un salto de línea podría inyectar cabeceras falsas: queda codificado.
        ("hola\r\nX-Admin: true", "hola%0D%0AX-Admin: true"),
    ],
)
def test_header_values_are_ascii_safe(value: str | float | bool, expected: str) -> None:
    encoded = encode_header_value(value)

    assert encoded == expected
    assert encoded.isascii()
    assert "\n" not in encoded
    assert "\r" not in encoded


@pytest.mark.parametrize(
    "text",
    ["Quiero información sobre mi factura de 50 €", "100% seguro", "%41", "  a\tb  "],
)
def test_text_header_value_round_trips(text: str) -> None:
    assert unquote(encode_header_value(text)) == text
