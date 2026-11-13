"""Validación y normalización de los datos que devuelve el ParserModel.

El ParserModel (en producción, un LLM) puede devolver cualquier cosa: formatos
raros, tipos equivocados, textos vacíos... Antes de decidir si se ejecuta una
acción, cada valor pasa por estas funciones, que:

- devuelven el valor normalizado si es válido,
- devuelven None si el dato todavía no se ha obtenido,
- lanzan `InvalidDataError` si el dato está, pero es incorrecto.

Son funciones puras (sin estado), así que son fáciles de probar y reutilizar
en agentes nuevos.
"""

import math
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# [0-9] en lugar de \d: \d también acepta dígitos de otros alfabetos (p. ej. "٣").
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

# Importe en texto: "150", "150.5", "150,50". Como mucho 2 decimales, porque
# "1.500" es ambiguo (¿1,5 € o 1.500 €?) y preferimos rechazarlo a adivinar.
_AMOUNT_TEXT = re.compile(r"[0-9]+(?:[.,][0-9]{1,2})?")

_CENT = Decimal("0.01")

# Tope de cordura: ningún compromiso de pago real se acerca. Además, por debajo
# de él el float conserva los céntimos exactos y se escribe sin notación
# científica (1e16 se escribiría "1e+16" en el body y en la cabecera).
_MAX_AMOUNT = Decimal("1000000000")  # mil millones


class InvalidDataError(ValueError):
    """Un dato extraído de la conversación existe pero no es válido."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


def optional_iso_date(value: object, *, field: str, not_before: date | None = None) -> str | None:
    """Valida una fecha 'yyyy-mm-dd'.

    - None o texto vacío → None (aún no hay fecha).
    - Debe ser una fecha real del calendario (rechaza "2026-02-30").
    - Si se indica `not_before`, no puede ser anterior a esa fecha.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidDataError(
            field, f"se esperaba un texto 'yyyy-mm-dd', llegó {type(value).__name__}"
        )

    text = value.strip()
    if not text:
        return None
    # Comprobamos el formato exacto antes de convertir: date.fromisoformat
    # también acepta otros formatos ISO (como "20261120") que no queremos.
    if not _ISO_DATE.fullmatch(text):
        raise InvalidDataError(field, f"formato no válido {text!r}, se esperaba 'yyyy-mm-dd'")
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise InvalidDataError(field, f"la fecha {text!r} no existe") from None

    if not_before is not None and parsed < not_before:
        raise InvalidDataError(
            field, f"la fecha {text} ya ha pasado (no puede ser anterior a {not_before})"
        )
    return parsed.isoformat()


def optional_positive_amount(value: object, *, field: str) -> float | None:
    """Valida un importe positivo y lo redondea a céntimos.

    - None o texto vacío → None (aún no hay importe).
    - Acepta números y textos como "150,50". Rechaza booleanos, NaN e infinito.
    - Redondea a 2 decimales (half-up, como se redondea el dinero).
    - Exige que sea mayor que 0 y no supere mil millones.
    """
    if value is None:
        return None
    # bool es subclase de int en Python: sin esta comprobación, True valdría 1 €.
    if isinstance(value, bool):
        raise InvalidDataError(field, "se esperaba un número, llegó un booleano")

    if isinstance(value, int):
        # Directo, sin pasar por texto: es exacto, y str() falla con enteros de
        # más de 4300 cifras.
        amount = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidDataError(field, "el importe debe ser un número finito")
        # str() da la representación decimal más corta del float (150.1 → "150.1"),
        # así evitamos arrastrar el error binario de coma flotante a Decimal.
        amount = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not _AMOUNT_TEXT.fullmatch(text):
            raise InvalidDataError(field, f"formato de importe no válido {text!r}")
        amount = Decimal(text.replace(",", "."))
    else:
        raise InvalidDataError(field, f"se esperaba un número, llegó {type(value).__name__}")

    try:
        rounded = amount.quantize(_CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        # Números tan grandes que no caben en la precisión de Decimal.
        raise InvalidDataError(field, "el importe está fuera de rango") from None

    if rounded <= 0:
        raise InvalidDataError(field, "el importe debe ser mayor que 0")
    if rounded > _MAX_AMOUNT:
        raise InvalidDataError(field, "el importe supera el máximo de mil millones")
    return float(rounded)


def optional_text(value: object, *, field: str) -> str | None:
    """Valida un texto libre: quita espacios sobrantes; vacío → None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidDataError(field, f"se esperaba un texto, llegó {type(value).__name__}")
    text = value.strip()
    return text or None
