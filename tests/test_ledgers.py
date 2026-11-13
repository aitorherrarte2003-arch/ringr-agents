"""Tests de los registros anti-duplicados (`ActionLedger`).

Los tests del contrato se ejecutan contra las dos implementaciones: si las dos
lo cumplen, el agente puede usar cualquiera de ellas sin cambiar nada.
"""

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from ringr_agents.agents import DebtAgent
from ringr_agents.conversation import Conversation
from ringr_agents.http_client import HttpRequest, HttpResponse, SimulatedHttpClient
from ringr_agents.idempotency import (
    ActionLedger,
    InMemoryActionLedger,
    Reservation,
    SqliteActionLedger,
)
from ringr_agents.results import ActionOutcome
from tests.helpers import build_agent

COMPLETE = {"commitment_date": "2026-11-20", "committed_amount": 150.5}


@pytest.fixture(params=["memoria", "sqlite"])
def ledger(request: pytest.FixtureRequest, tmp_path: Path) -> ActionLedger:
    """Cada test que pide `ledger` se ejecuta dos veces, una con cada registro."""
    if request.param == "memoria":
        return InMemoryActionLedger()
    return SqliteActionLedger(tmp_path / "acciones.db")


def reserved(ledger: ActionLedger, fingerprint: str) -> Reservation:
    """Reserva y comprueba que se ha concedido."""
    reservation = ledger.reserve("scope", fingerprint)
    assert reservation is not None
    return reservation


class TestContract:
    def test_the_first_send_is_revision_1(self, ledger: ActionLedger) -> None:
        assert ledger.reserve("scope", "A") == Reservation("scope", "A", revision=1)

    def test_a_send_in_progress_blocks_any_other(self, ledger: ActionLedger) -> None:
        reserved(ledger, "A")

        assert ledger.reserve("scope", "A") is None
        assert ledger.reserve("scope", "B") is None

    def test_the_last_data_sent_is_a_duplicate(self, ledger: ActionLedger) -> None:
        ledger.confirm(reserved(ledger, "A"))

        assert ledger.reserve("scope", "A") is None

    def test_new_data_is_the_next_revision(self, ledger: ActionLedger) -> None:
        ledger.confirm(reserved(ledger, "A"))

        assert ledger.reserve("scope", "B") == Reservation("scope", "B", revision=2)

    def test_going_back_to_earlier_data_is_a_new_send(self, ledger: ActionLedger) -> None:
        ledger.confirm(reserved(ledger, "A"))
        ledger.confirm(reserved(ledger, "B"))

        assert ledger.reserve("scope", "A") == Reservation("scope", "A", revision=3)

    def test_a_released_send_can_be_retried_with_the_same_key(self, ledger: ActionLedger) -> None:
        first = reserved(ledger, "A")
        ledger.release(first)

        retry = reserved(ledger, "A")
        assert retry == first
        assert retry.idempotency_key == first.idempotency_key

    def test_only_the_owner_can_confirm_or_release(self, ledger: ActionLedger) -> None:
        owner = reserved(ledger, "A")
        stranger = Reservation("scope", "A", revision=1)  # misma reserva, otro token

        ledger.release(stranger)
        assert ledger.reserve("scope", "B") is None  # sigue reservado por su dueño
        ledger.confirm(stranger)
        ledger.release(owner)
        assert ledger.reserve("scope", "A") is not None  # nadie confirmó "A"

    def test_scopes_are_independent(self, ledger: ActionLedger) -> None:
        reserved(ledger, "A")

        assert ledger.reserve("other-scope", "A") == Reservation("other-scope", "A", revision=1)

    def test_only_one_of_many_simultaneous_reservations_wins(self, ledger: ActionLedger) -> None:
        threads_count = 8
        start = threading.Barrier(threads_count)  # todos los hilos arrancan a la vez
        results: list[Reservation | None] = []

        def worker() -> None:
            start.wait()
            results.append(ledger.reserve("scope", "A"))

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == threads_count
        assert sum(result is not None for result in results) == 1


class TestInMemory:
    def test_the_lock_makes_check_and_reserve_a_single_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Entre "comprobar que está libre" y "apuntar la reserva", `reserve` crea
        # el token con uuid.uuid4(). Lo hacemos lento a propósito para agrandar
        # ese hueco: sin el cerrojo, los 8 hilos comprobarían "libre" a la vez y
        # los 8 reservarían. Con él, cada hilo espera a que termine el anterior.
        real_uuid4 = uuid.uuid4

        def slow_uuid4() -> uuid.UUID:
            time.sleep(0.01)
            return real_uuid4()

        monkeypatch.setattr(uuid, "uuid4", slow_uuid4)
        ledger = InMemoryActionLedger()
        start = threading.Barrier(8)
        results: list[Reservation | None] = []

        def worker() -> None:
            start.wait()
            results.append(ledger.reserve("scope", "A"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(result is not None for result in results) == 1


class TestSqlite:
    def test_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "acciones.db"
        first_process = SqliteActionLedger(path)
        first_process.confirm(reserved(first_process, "A"))

        restarted = SqliteActionLedger(path)

        assert restarted.reserve("scope", "A") is None
        assert restarted.reserve("scope", "B") == Reservation("scope", "B", revision=2)

    def test_an_abandoned_reservation_expires(self, tmp_path: Path) -> None:
        now = 1000.0

        def clock() -> float:
            return now

        ledger = SqliteActionLedger(tmp_path / "a.db", stale_after_seconds=60, clock=clock)
        reserved(ledger, "A")  # el proceso "muere" aquí y nunca confirma ni libera

        now = 1059.0
        assert ledger.reserve("scope", "A") is None  # todavía puede estar enviándose
        now = 1060.0
        assert ledger.reserve("scope", "A") == Reservation("scope", "A", revision=1)

    def test_a_confirmed_send_never_expires(self, tmp_path: Path) -> None:
        now = 1000.0

        def clock() -> float:
            return now

        ledger = SqliteActionLedger(tmp_path / "a.db", stale_after_seconds=60, clock=clock)
        ledger.confirm(reserved(ledger, "A"))

        now = 10.0**9
        assert ledger.reserve("scope", "A") is None

    def test_a_late_owner_cannot_touch_a_reclaimed_reservation(self, tmp_path: Path) -> None:
        now = 1000.0

        def clock() -> float:
            return now

        ledger = SqliteActionLedger(tmp_path / "a.db", stale_after_seconds=60, clock=clock)
        slow = reserved(ledger, "A")  # se queda colgado más de 60 s
        now = 1100.0
        rescuer = reserved(ledger, "B")  # caducó: otro turno la recupera, con datos nuevos

        ledger.release(slow)  # llega tarde: no puede liberar la reserva de otro
        assert ledger.reserve("scope", "B") is None
        ledger.confirm(rescuer)
        ledger.confirm(slow)  # tampoco puede confirmar la suya, que ya no existe
        assert ledger.reserve("scope", "A") == Reservation("scope", "A", revision=2)

    @pytest.mark.parametrize("seconds", [0, -1])
    def test_expiry_must_be_positive(self, tmp_path: Path, seconds: float) -> None:
        with pytest.raises(ValueError, match="mayor que 0"):
            SqliteActionLedger(tmp_path / "a.db", stale_after_seconds=seconds)

    def test_memory_database_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="necesita un fichero"):
            SqliteActionLedger(":memory:")

    def test_missing_folders_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "datos" / "registro" / "acciones.db"
        SqliteActionLedger(path)
        assert path.exists()

    def test_an_old_sqlite_is_rejected_with_a_clear_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 31, 1))
        with pytest.raises(RuntimeError, match=r"3\.35"):
            SqliteActionLedger(tmp_path / "a.db")


class TestWithAgents:
    def test_a_restarted_service_does_not_repeat_the_action(self, tmp_path: Path) -> None:
        path = tmp_path / "acciones.db"
        client = SimulatedHttpClient()

        def start_service() -> DebtAgent:
            # Mismo id de llamada y mismo fichero: como si el servicio se reiniciara.
            return build_agent(
                DebtAgent,
                [COMPLETE],
                http_client=client,
                ledger=SqliteActionLedger(path),
                conversation=Conversation("llamada-42"),
            ).agent

        before = start_service().handle_turn("150,50 el 20")
        after_restart = start_service().handle_turn("150,50 el 20")

        assert before.outcome is ActionOutcome.EXECUTED
        assert after_restart.outcome is ActionOutcome.DUPLICATE
        assert len(client.sent_requests) == 1

    def test_an_unexpected_error_frees_the_reservation(self, ledger: ActionLedger) -> None:
        calls: list[HttpRequest] = []

        def broken_then_ok(request: HttpRequest) -> HttpResponse:
            calls.append(request)
            if len(calls) == 1:
                raise RuntimeError("fallo inesperado")
            return HttpResponse(200)

        harness = build_agent(
            DebtAgent,
            [COMPLETE, COMPLETE],
            http_client=SimulatedHttpClient(broken_then_ok),
            ledger=ledger,
        )

        with pytest.raises(RuntimeError, match="inesperado"):
            harness.agent.handle_turn("150,50 el 20")
        retried = harness.agent.handle_turn("¿Sigue ahí?")

        assert retried.outcome is ActionOutcome.EXECUTED
        assert calls[0].headers["Idempotency-Key"] == calls[1].headers["Idempotency-Key"]


class _BrokenLedger:
    """Registro que falla en el paso indicado (simula, p. ej., "database is locked")."""

    def __init__(self, failing_step: str) -> None:
        self._inner = InMemoryActionLedger()
        self._failing_step = failing_step

    def reserve(self, scope: str, fingerprint: str) -> Reservation | None:
        self._fail_if("reserve")
        return self._inner.reserve(scope, fingerprint)

    def confirm(self, reservation: Reservation) -> None:
        self._fail_if("confirm")
        self._inner.confirm(reservation)

    def release(self, reservation: Reservation) -> None:
        self._fail_if("release")
        self._inner.release(reservation)

    def _fail_if(self, step: str) -> None:
        if step == self._failing_step:
            raise sqlite3.OperationalError("database is locked")


class TestBrokenLedger:
    def test_if_the_ledger_cannot_reserve_nothing_is_sent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = build_agent(DebtAgent, [COMPLETE], ledger=_BrokenLedger("reserve"))

        with caplog.at_level(logging.WARNING, logger="ringr_agents"):
            result = harness.agent.handle_turn("150,50 el 20")

        assert result.outcome is ActionOutcome.FAILED
        assert result.reply == "Respuesta 1"  # el cliente tiene su respuesta
        assert harness.http_client.sent_requests == ()
        assert "cause=ledger_error" in caplog.text

    def test_if_the_ledger_cannot_confirm_the_turn_still_finishes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness = build_agent(DebtAgent, [COMPLETE], ledger=_BrokenLedger("confirm"))

        with caplog.at_level(logging.WARNING, logger="ringr_agents"):
            result = harness.agent.handle_turn("150,50 el 20")

        assert result.outcome is ActionOutcome.EXECUTED  # el envío ya se hizo
        assert len(harness.http_client.sent_requests) == 1
        assert "cause=ledger_error" in caplog.text

    def test_a_ledger_failure_never_hides_the_original_error(self) -> None:
        def broken(request: HttpRequest) -> HttpResponse:
            raise RuntimeError("fallo inesperado")

        harness = build_agent(
            DebtAgent,
            [COMPLETE],
            http_client=SimulatedHttpClient(broken),
            ledger=_BrokenLedger("release"),
        )

        with pytest.raises(RuntimeError, match="inesperado"):
            harness.agent.handle_turn("150,50 el 20")
