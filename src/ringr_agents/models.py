"""Modelos de lenguaje: los objetos genéricos del enunciado y sus contratos.

El enunciado describe dos objetos genéricos:

- `ConversationModel`, con `answer_user()`, que devuelve un `str`: la respuesta
  que el agente dice al usuario.
- `ParserModel`, con `parse_data()`, que devuelve un `dict` tipo JSON: los datos
  extraídos de la conversación.

Y dice que llevan "en su estado interno" toda la información para adaptarse al
caso de uso. Por eso sus métodos no reciben argumentos: lo que necesitan lo
tienen dentro. Aquí están simulados (el enunciado pide no llamar a LLMs reales)
y su estado es un guion de respuestas. En `llm/claude.py` hay versiones reales,
con Claude, cuyo estado es la conversación, las instrucciones y el esquema.

Los agentes no dependen de estas clases, sino de los contratos
(`ConversationModelProtocol` y `ParserModelProtocol`): cualquier objeto con el
método correspondiente sirve, sin heredar de nada.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Protocol


class ConversationModelProtocol(Protocol):
    """Contrato: genera la respuesta que el agente dice al usuario."""

    def answer_user(self) -> str: ...


class ParserModelProtocol(Protocol):
    """Contrato: extrae datos estructurados (un dict tipo JSON) de la conversación."""

    def parse_data(self) -> dict[str, Any]: ...


class ModelResponseError(RuntimeError):
    """El modelo no dio una respuesta útil (rechazo, texto cortado, JSON roto, o la
    API no respondió). Lo lanza cualquier implementación real de los contratos."""


class ModelUnavailableError(ModelResponseError):
    """La API del modelo no respondió (sin conexión, saturada, 5xx).

    Es un `ModelResponseError`, así que el agente lo gestiona igual. Las
    evaluaciones, en cambio, se paran: no es un fallo de extracción.
    """


class _Script[T]:
    """Valores en orden, uno por llamada. Al llegar al último, lo sigue repitiendo."""

    def __init__(self, values: tuple[T, ...], *, owner: str) -> None:
        if not values:
            raise ValueError(f"{owner} necesita al menos un valor en su guion.")
        self._values = values
        self._position = 0

    def next(self) -> T:
        value = self._values[self._position]
        if self._position < len(self._values) - 1:
            self._position += 1
        return value


class ConversationModel:
    """ConversationModel genérico y simulado: su estado es un guion de respuestas.

        ConversationModel("Anotado, gracias.")                    # siempre la misma
        ConversationModel(["¿Cuándo podría pagar?", "Anotado."])  # una por turno

    Al acabarse el guion repite la última respuesta: se puede llamar a
    `handle_turn` tantas veces como se quiera.
    """

    def __init__(self, replies: str | Iterable[str]) -> None:
        # Un texto suelto es una sola respuesta (sin esto, se tomaría letra a letra).
        values = (replies,) if isinstance(replies, str) else tuple(replies)
        self._script = _Script(values, owner="ConversationModel")

    def answer_user(self) -> str:
        return self._script.next()


class ParserModel:
    """ParserModel genérico y simulado: su estado es un guion de resultados.

        ParserModel({"request": "Darme de baja"})               # siempre el mismo
        ParserModel([{"request": None}, {"request": "Baja"}])   # uno por turno

    Al acabarse el guion repite el último resultado, como un parser real que
    vuelve a leer una conversación que ya no cambia.
    """

    def __init__(self, outputs: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
        # Un dict suelto es un solo resultado. Se copian: si quien los pasó los
        # modifica después, el guion no cambia.
        values = (outputs,) if isinstance(outputs, Mapping) else tuple(outputs)
        self._script = _Script(tuple(dict(value) for value in values), owner="ParserModel")

    def parse_data(self) -> dict[str, Any]:
        # Otra copia: si el agente (o un test) modifica el resultado, el guion no cambia.
        return dict(self._script.next())
