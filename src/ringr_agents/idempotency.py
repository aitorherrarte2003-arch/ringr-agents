"""Evitar acciones duplicadas.

El ParserModel lee la conversación entera en cada turno. Si el usuario dijo
"pago 150 € el día 20" en el turno 3, en los turnos 4, 5 y 6 el parser seguirá
devolviendo esos mismos datos. Sin control, registraríamos el compromiso
cuatro veces.

Cada acción de cada conversación es un "ámbito" (scope). El registro
(`ActionLedger`) guarda, por ámbito, la huella de los últimos datos enviados
con éxito y cuántos envíos lleva. Un turno solo envía si sus datos son
distintos de los últimos enviados:

    150 € → se envía · 150 € → duplicado · 200 € → se envía · 150 € → se envía

Antes de enviar, el agente reserva el ámbito: comprobar y apuntar en un solo
paso, para que dos turnos simultáneos no envíen los dos. Si el envío sale
bien, la reserva se confirma; si falla, se libera y un turno posterior lo
reintentará.

Cada envío lleva en la cabecera `Idempotency-Key` una clave única de ese
envío, la práctica estándar para que el servidor también pueda descartar
duplicados (por ejemplo, si un reintento llega dos veces).
"""

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ringr_agents.http_client import JsonScalar


def action_scope(conversation_id: str, action_name: str) -> str:
    """Identifica "esta acción en esta conversación".

    Incluye el id de la conversación para que dos clientes distintos que se
    comprometan a lo mismo no se confundan entre sí.
    """
    return _sha256({"conversation_id": conversation_id, "action": action_name})


def data_fingerprint(payload: Mapping[str, JsonScalar]) -> str:
    """Huella determinista de los datos: mismos datos → misma huella."""
    return _sha256(dict(payload))


def _sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,  # el orden de las claves no debe cambiar la huella
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Reservation:
    """Permiso para enviar una acción. Lo da `ActionLedger.reserve`."""

    scope: str
    fingerprint: str
    # Número de este envío dentro del ámbito: 1 el primero, 2 si los datos
    # cambian una vez, etc. Un reintento tras un fallo repite el número.
    revision: int
    # Identifica a quien tiene la reserva: solo él puede confirmarla o liberarla.
    # No cuenta al comparar dos reservas ni forma parte de la clave.
    token: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)

    @property
    def idempotency_key(self) -> str:
        """Clave de este envío. Un reintento con los mismos datos repite la clave;
        unos datos nuevos (aunque vuelvan a un valor anterior) llevan otra."""
        return _sha256([self.scope, self.revision, self.fingerprint])


class ActionLedger(Protocol):
    """Registro de envíos: por cada ámbito, los últimos datos enviados con éxito."""

    def reserve(self, scope: str, fingerprint: str) -> Reservation | None:
        """Reserva el envío, de forma atómica.

        Devuelve None si no hay que enviar: los datos son los últimos enviados
        (duplicado) o hay otro envío en curso en ese ámbito.
        """
        ...

    def confirm(self, reservation: Reservation) -> None:
        """El envío salió bien: sus datos pasan a ser los últimos enviados.

        Si la reserva ya no es suya (caducó y otro la recuperó), no hace nada.
        """
        ...

    def release(self, reservation: Reservation) -> None:
        """El envío falló: el ámbito queda libre para reintentarlo.

        Si la reserva ya no es suya, no hace nada: nunca libera la de otro.
        """
        ...


@dataclass
class _ScopeState:
    fingerprint: str | None = None  # huella del último envío confirmado
    revision: int = 0  # cuántos envíos confirmados lleva el ámbito
    reserved_by: str | None = None  # token del envío en curso (None: ninguno)


class InMemoryActionLedger:
    """Registro en memoria: vale para un proceso y se pierde al reiniciar.

    Un cerrojo (`threading.Lock`) hace que comprobar y apuntar sea un solo
    paso aunque varios hilos atiendan turnos a la vez.
    """

    def __init__(self) -> None:
        self._scopes: dict[str, _ScopeState] = {}
        self._lock = threading.Lock()

    def reserve(self, scope: str, fingerprint: str) -> Reservation | None:
        with self._lock:
            state = self._scopes.setdefault(scope, _ScopeState())
            if state.reserved_by is not None or state.fingerprint == fingerprint:
                return None
            reservation = Reservation(scope, fingerprint, state.revision + 1)
            state.reserved_by = reservation.token
            return reservation

    def confirm(self, reservation: Reservation) -> None:
        with self._lock:
            state = self._scopes.get(reservation.scope)
            if state is None or state.reserved_by != reservation.token:
                return  # la reserva no es suya
            state.fingerprint = reservation.fingerprint
            state.revision = reservation.revision
            state.reserved_by = None

    def release(self, reservation: Reservation) -> None:
        with self._lock:
            state = self._scopes.get(reservation.scope)
            if state is not None and state.reserved_by == reservation.token:
                state.reserved_by = None


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS actions (
    scope       TEXT PRIMARY KEY,
    fingerprint TEXT,              -- huella del último envío confirmado (NULL: ninguno)
    revision    INTEGER NOT NULL,  -- cuántos envíos confirmados lleva el ámbito
    reserved_by TEXT,              -- token del envío en curso (NULL: ninguno)
    reserved_at REAL               -- cuándo se reservó ese envío
)
"""

# Una sola sentencia: crea la fila (primer envío) o la reserva si está libre
# (o su reserva caducó) y los datos son distintos de los últimos enviados.
# Si no se cumple, no cambia nada y no devuelve ninguna fila.
_RESERVE = """
INSERT INTO actions (scope, revision, reserved_by, reserved_at)
VALUES (:scope, 0, :token, :now)
ON CONFLICT (scope) DO UPDATE SET reserved_by = :token, reserved_at = :now
WHERE (actions.reserved_by IS NULL OR actions.reserved_at <= :expired)
  AND actions.fingerprint IS NOT :fingerprint
RETURNING revision
"""

# Las dos solo cambian la fila si la reserva sigue siendo de quien llama.
_CONFIRM = """
UPDATE actions
SET fingerprint = :fingerprint, revision = :revision, reserved_by = NULL, reserved_at = NULL
WHERE scope = :scope AND reserved_by = :token
"""

_RELEASE = """
UPDATE actions SET reserved_by = NULL, reserved_at = NULL
WHERE scope = :scope AND reserved_by = :token
"""

# RETURNING existe desde SQLite 3.35 (2021). Python usa el SQLite del sistema en
# Linux, y algunas distribuciones antiguas traen uno anterior.
_MIN_SQLITE = (3, 35, 0)


class SqliteActionLedger:
    """Registro en un fichero SQLite (incluido en Python): sobrevive a los
    reinicios y lo pueden compartir varios procesos de la misma máquina.

    Si el proceso muere entre reservar y confirmar, la reserva se quedaría
    colgada y la acción no se enviaría nunca. Por eso una reserva caduca a
    los `stale_after_seconds`, que deben superar lo que tarda un envío con
    sus reintentos. Si aun así llegaran dos envíos, el servidor los descarta
    por la cabecera Idempotency-Key.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stale_after_seconds: float = 120.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds debe ser mayor que 0.")
        if str(path) == ":memory:":
            # Cada conexión abriría una base de datos vacía distinta.
            raise ValueError("SqliteActionLedger necesita un fichero, no ':memory:'.")
        if sqlite3.sqlite_version_info < _MIN_SQLITE:
            raise RuntimeError(
                f"SqliteActionLedger necesita SQLite 3.35 o superior; hay {sqlite3.sqlite_version}."
            )
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stale_after = stale_after_seconds
        # Reloj inyectable: en los tests se "adelanta el tiempo" sin esperar.
        self._clock = clock
        with self._connect() as db:
            db.execute(_CREATE_TABLE)

    def reserve(self, scope: str, fingerprint: str) -> Reservation | None:
        now = self._clock()
        token = uuid.uuid4().hex
        params = {
            "scope": scope,
            "fingerprint": fingerprint,
            "token": token,
            "now": now,
            "expired": now - self._stale_after,
        }
        # SQLite ejecuta las escrituras de una en una: aunque dos procesos
        # lleguen a la vez, solo uno consigue la fila.
        with self._connect() as db:
            rows = db.execute(_RESERVE, params).fetchall()
        if not rows:
            return None
        return Reservation(scope, fingerprint, rows[0][0] + 1, token)

    def confirm(self, reservation: Reservation) -> None:
        params = {
            "scope": reservation.scope,
            "fingerprint": reservation.fingerprint,
            "revision": reservation.revision,
            "token": reservation.token,
        }
        with self._connect() as db:
            db.execute(_CONFIRM, params)

    def release(self, reservation: Reservation) -> None:
        params = {"scope": reservation.scope, "token": reservation.token}
        with self._connect() as db:
            db.execute(_RELEASE, params)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Abre una conexión, guarda los cambios al terminar y la cierra siempre."""
        connection = sqlite3.connect(self._path, timeout=5.0)
        try:
            with connection:  # guarda si todo va bien; deshace si hay un error
                yield connection
        finally:
            connection.close()
