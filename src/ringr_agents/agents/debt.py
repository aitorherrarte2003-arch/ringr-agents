"""Agente de cobros: registra compromisos de pago de deudas pendientes."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Self

from ringr_agents.actions import HttpAction
from ringr_agents.agents.base import BaseAgent
from ringr_agents.http_client import JsonScalar
from ringr_agents.validation import optional_iso_date, optional_positive_amount


@dataclass(frozen=True, slots=True)
class DebtCommitment:
    """Compromiso de pago ya validado. Cada campo es None si aún no se conoce."""

    commitment_date: str | None  # 'yyyy-mm-dd'
    committed_amount: float | None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], *, today: date) -> Self:
        """Valida y normaliza la salida del ParserModel.

        Lanza `InvalidDataError` si algún dato es incorrecto.
        """
        return cls(
            # Un compromiso de pago con fecha pasada no tiene sentido: se rechaza.
            commitment_date=optional_iso_date(
                raw.get("commitment_date"), field="commitment_date", not_before=today
            ),
            committed_amount=optional_positive_amount(
                raw.get("committed_amount"), field="committed_amount"
            ),
        )

    def to_payload(self) -> Mapping[str, JsonScalar]:
        return {
            "commitment_date": self.commitment_date,
            "committed_amount": self.committed_amount,
        }


class DebtAgent(BaseAgent[DebtCommitment]):
    """Registra el compromiso cuando hay fecha e importe."""

    action = HttpAction(name="debt.commitment", url="https://api.ringr.debt/v1/commitment")

    def validate_data(self, raw: Mapping[str, Any]) -> DebtCommitment:
        return DebtCommitment.from_raw(raw, today=self._today())

    def should_act(self, data: DebtCommitment) -> bool:
        return data.commitment_date is not None and data.committed_amount is not None
