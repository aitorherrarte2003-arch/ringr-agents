"""Agente base: el ciclo de vida de un turno, común a todos los agentes.

Patrón "template method": `handle_turn` fija los pasos y su orden, que son
iguales para cualquier agente. Cada agente concreto solo rellena las partes
que cambian:

    action          → a qué integración llama
    validate_data() → cómo valida y normaliza lo que extrae el ParserModel
    should_act()    → cuándo hay datos suficientes para actuar

Y, solo si lo necesita, `dedup_payload()`: qué datos cuentan para decidir si
una acción es nueva o repetida (por defecto, todos los que se envían).

Todo lo demás (respuesta al usuario, anti-duplicados, construcción y envío de
la petición, gestión de errores) está aquí una sola vez.

Las dependencias (modelos, cliente HTTP, autenticación, registro de acciones,
reloj) se reciben en el constructor en lugar de crearse dentro. Esto se llama
inyección de dependencias y es lo que permite testear el agente con objetos
simulados o cambiar el LLM sin tocar este código. Solo los dos modelos son
obligatorios: lo demás tiene el valor que pide el enunciado (HTTP simulado y
el Bearer token de pruebas), así que basta con `DebtAgent(conversation_model,
parser_model)`.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, Protocol

from ringr_agents.actions import HttpAction
from ringr_agents.auth import Authenticator, BearerTokenAuth
from ringr_agents.config import Settings
from ringr_agents.conversation import Conversation
from ringr_agents.http_client import (
    HttpClient,
    HttpTransportError,
    JsonScalar,
    SimulatedHttpClient,
)
from ringr_agents.idempotency import (
    ActionLedger,
    InMemoryActionLedger,
    Reservation,
    action_scope,
    data_fingerprint,
)
from ringr_agents.models import (
    ConversationModelProtocol,
    ModelResponseError,
    ParserModelProtocol,
)
from ringr_agents.results import ActionOutcome, TurnResult
from ringr_agents.validation import InvalidDataError

logger = logging.getLogger(__name__)


class AgentData(Protocol):
    """Lo mínimo que deben cumplir los datos validados de cualquier agente."""

    def to_payload(self) -> Mapping[str, JsonScalar]:
        """Datos tal y como se envían a la API (body y cabeceras)."""
        ...


class BaseAgent[DataT: AgentData](ABC):
    """Base de todos los agentes. No se instancia directamente."""

    def __init__(
        self,
        conversation_model: ConversationModelProtocol,
        parser_model: ParserModelProtocol,
        *,
        http_client: HttpClient | None = None,
        auth: Authenticator | None = None,
        ledger: ActionLedger | None = None,
        conversation: Conversation | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._conversation_model = conversation_model
        self._parser_model = parser_model
        # Por defecto, lo que pide el enunciado: peticiones simuladas (200 OK) y
        # el Bearer token de pruebas (o el de RINGR_API_TOKEN, si está definida).
        self._http_client = http_client if http_client is not None else SimulatedHttpClient()
        self._auth = (
            auth if auth is not None else BearerTokenAuth.from_settings(Settings.from_env())
        )
        self._ledger = ledger if ledger is not None else InMemoryActionLedger()
        self._conversation = conversation if conversation is not None else Conversation()
        # Inyectar el reloj permite fijar "hoy" en los tests (fechas pasadas/futuras).
        self._today = today
        # Contenido de los envíos que fallaron, por Idempotency-Key: un reintento
        # reenvía exactamente lo mismo (un servidor puede rechazar una clave
        # repetida con otro contenido, y el LLM puede redactar distinto).
        self._failed_payloads: dict[str, Mapping[str, JsonScalar]] = {}
        self._turns = 0

    # --- Lo que cada agente concreto debe definir ---------------------------

    @property
    @abstractmethod
    def action(self) -> HttpAction:
        """Integración que ejecuta este agente."""

    @abstractmethod
    def validate_data(self, raw: Mapping[str, Any]) -> DataT:
        """Valida y normaliza los datos del ParserModel.

        Lanza `InvalidDataError` si algún dato es incorrecto.
        """

    @abstractmethod
    def should_act(self, data: DataT) -> bool:
        """Regla de negocio: ¿hay datos suficientes para ejecutar la acción?"""

    # --- Lo que cada agente puede cambiar si lo necesita --------------------

    def dedup_payload(self, data: DataT) -> Mapping[str, JsonScalar]:
        """Qué datos deciden si una acción es nueva o repetida.

        Por defecto, todos los que se envían: si cambian, se vuelve a enviar.
        """
        return data.to_payload()

    # --- Ciclo de vida del turno (igual para todos los agentes) -------------

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    def handle_turn(self, user_message: str | None = None) -> TurnResult[DataT]:
        """Ejecuta un turno y devuelve lo que ha ocurrido.

        Tal y como lo describe el enunciado, un turno no necesita argumentos:
        los modelos ya tienen en su estado lo que necesitan. `user_message` es
        opcional: si se pasa (lo hace el chat de la CLI), se apunta antes en la
        conversación, que es de donde lo leen los modelos reales.
        """
        self._turns += 1

        # 1. Responder al usuario.
        if user_message is not None:
            self._conversation.add_user_message(user_message)
        reply = self._conversation_model.answer_user()
        self._conversation.add_assistant_message(reply)

        # 2 y 3. Extraer datos de la conversación (incluida la respuesta recién
        # dada), validarlos y normalizarlos. Si el parser falla o devuelve datos
        # incorrectos, el usuario ya tiene su respuesta: el turno no se rompe,
        # solo no se actúa.
        try:
            raw = self._parser_model.parse_data()
            data = self._validate(raw)
        except (ModelResponseError, InvalidDataError) as error:
            # En el log va el campo que falló, nunca el valor rechazado.
            field = error.field if isinstance(error, InvalidDataError) else "parser_model"
            self._warn(f"invalid_data field={field}")
            return self._finish(
                TurnResult(reply=reply, outcome=ActionOutcome.INVALID_DATA, error=str(error))
            )

        # 4. Decidir si se actúa.
        if not self.should_act(data):
            return self._finish(TurnResult(reply=reply, outcome=ActionOutcome.NOT_READY, data=data))

        # 5. Ejecutar la acción (sin duplicados).
        return self._finish(self._execute_action(reply, data))

    # --- Detalles internos ---------------------------------------------------

    def _validate(self, raw: object) -> DataT:
        # El contrato dice dict, pero si el modelo devuelve otra cosa no
        # queremos que el agente se rompa: lo tratamos como dato inválido.
        if not isinstance(raw, Mapping):
            raise InvalidDataError(
                "parser_output", f"se esperaba un dict, llegó {type(raw).__name__}"
            )
        return self.validate_data(raw)

    def _execute_action(self, reply: str, data: DataT) -> TurnResult[DataT]:
        scope = action_scope(self._conversation.id, self.action.name)
        try:
            reservation = self._ledger.reserve(scope, data_fingerprint(self.dedup_payload(data)))
        except Exception:
            # Sin registro no se puede saber si sería un duplicado: no se envía.
            self._warn("ledger_error")
            return TurnResult(
                reply=reply,
                outcome=ActionOutcome.FAILED,
                data=data,
                error="No se pudo consultar el registro anti-duplicados",
            )
        if reservation is None:  # ya enviado, o hay otro envío en curso
            return TurnResult(reply=reply, outcome=ActionOutcome.DUPLICATE, data=data)

        key = reservation.idempotency_key
        payload = self._failed_payloads.pop(key, data.to_payload())
        try:
            result = self._send(reply, data, payload, key)
        except BaseException:
            # Un error inesperado no puede dejar la reserva colgada: se libera
            # para que otro turno lo reintente (si el envío sí llegó, el
            # servidor lo descarta por la Idempotency-Key) y el error sigue.
            self._failed_payloads[key] = payload
            self._settle(self._ledger.release, reservation)
            raise

        # Solo se confirma si tuvo éxito: si falla, se libera y el siguiente
        # turno lo reintentará con la misma Idempotency-Key.
        if result.action_executed:
            self._settle(self._ledger.confirm, reservation)
        else:
            self._failed_payloads[key] = payload
            self._settle(self._ledger.release, reservation)
        return result

    def _settle(self, step: Callable[[Reservation], None], reservation: Reservation) -> None:
        """Confirma o libera la reserva. Si el registro falla, el turno no se rompe:
        el envío ya se hizo (o falló) y no tiene vuelta atrás; queda el aviso en el log."""
        try:
            step(reservation)
        except Exception:
            self._warn("ledger_error")

    def _send(
        self, reply: str, data: DataT, payload: Mapping[str, JsonScalar], idempotency_key: str
    ) -> TurnResult[DataT]:
        request = self.action.build_request(
            payload, auth=self._auth, idempotency_key=idempotency_key
        )
        try:
            response = self._http_client.send(request)
        except HttpTransportError as error:
            self._warn("network_error")
            return TurnResult(
                reply=reply,
                outcome=ActionOutcome.FAILED,
                data=data,
                request=request,
                error=f"Error de red: {error}",
            )

        if not response.ok:
            self._warn(f"http_status={response.status_code}")
            return TurnResult(
                reply=reply,
                outcome=ActionOutcome.FAILED,
                data=data,
                request=request,
                response=response,
                error=f"El servidor respondió {response.status_code}",
            )

        return TurnResult(
            reply=reply,
            outcome=ActionOutcome.EXECUTED,
            data=data,
            request=request,
            response=response,
        )

    # Los logs nunca llevan datos personales (importes, fechas, textos del
    # cliente): solo qué agente, qué conversación y qué pasó.

    def _warn(self, cause: str) -> None:
        logger.warning(
            "agent=%s conversation=%s cause=%s",
            type(self).__name__,
            self._conversation.id,
            cause,
        )

    def _finish(self, result: TurnResult[DataT]) -> TurnResult[DataT]:
        logger.info(
            "agent=%s conversation=%s turn=%d outcome=%s",
            type(self).__name__,
            self._conversation.id,
            self._turns,
            result.outcome,
        )
        return result
