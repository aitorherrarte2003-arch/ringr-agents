"""Perfiles de agente: lo que un LLM real necesita saber de cada caso de uso.

El enunciado dice que los modelos genéricos "contienen en su estado interno
toda la información necesaria para adaptarse al caso de uso". Esa información
es esta: las instrucciones de conversación, las instrucciones de extracción y
el esquema JSON que debe cumplir la salida del parser.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentProfile:
    conversation_instructions: str
    parser_instructions: str
    parser_schema: dict[str, Any]


def _nullable(json_type: dict[str, Any]) -> dict[str, Any]:
    """Esquema JSON de "este tipo o null" (aún no se conoce el dato)."""
    return {"anyOf": [json_type, {"type": "null"}]}


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Objeto JSON con todos los campos obligatorios y ninguno extra."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_VOICE_STYLE = (
    "Hablas por teléfono: frases cortas, tono cercano y profesional, una sola pregunta "
    "cada vez. Responde siempre en español. Nunca inventes datos que el cliente no haya dado."
)

DEBT_PROFILE = AgentProfile(
    conversation_instructions=(
        "Eres un agente de cobros de Ringr. Llamas a un cliente con un recibo pendiente. "
        "Tu objetivo es acordar un compromiso de pago: una fecha concreta y un importe. "
        "Si el cliente da solo uno de los dos datos, pregunta por el otro. Cuando tengas "
        "ambos, confírmalos en voz alta y despídete. " + _VOICE_STYLE
    ),
    parser_instructions=(
        "Lee la conversación entre un agente de cobros y un cliente y extrae el compromiso "
        "de pago que el cliente ha aceptado.\n"
        "- commitment_date: fecha en la que el cliente se compromete a pagar, en formato "
        "yyyy-mm-dd. Resuelve fechas relativas ('el viernes', 'dentro de una semana') a "
        "partir de la fecha de hoy. null si el cliente no ha dado una fecha concreta.\n"
        "- committed_amount: importe en euros que el cliente se compromete a pagar, como "
        "número. null si no ha dado un importe concreto.\n"
        "Si el cliente cambia de opinión, usa el último valor que acepta. No uses importes "
        "o fechas que solo menciona el agente si el cliente no los ha aceptado."
    ),
    parser_schema=_object_schema(
        {
            "commitment_date": _nullable({"type": "string", "format": "date"}),
            "committed_amount": _nullable({"type": "number"}),
        }
    ),
)

ASSISTANCE_PROFILE = AgentProfile(
    conversation_instructions=(
        "Eres un agente de atención al cliente de Ringr. Resuelves dudas generales y, "
        "cuando el cliente pide algo que requiere a una persona (cambios de tarifa, bajas, "
        "reclamaciones, incidencias), le dices que registras su solicitud para que un "
        "compañero la gestione. " + _VOICE_STYLE
    ),
    parser_instructions=(
        "Lee la conversación entre un agente de atención al cliente y un cliente.\n"
        "- request: si el cliente pide algo que debe gestionar una persona (un cambio, una "
        "baja, una reclamación, una incidencia), resúmelo en una frase corta y concreta en "
        "español, por ejemplo 'Cambio de tarifa a fibra de 1 Gb'. null si el cliente solo "
        "hace preguntas que el agente puede contestar o todavía no ha pedido nada."
    ),
    parser_schema=_object_schema({"request": _nullable({"type": "string"})}),
)
