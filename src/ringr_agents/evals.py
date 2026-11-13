"""Evaluación del ParserModel con conversaciones de ejemplo.

Que el código compile no dice si el agente funciona con clientes reales. Una
evaluación pasa al parser conversaciones con la respuesta correcta ya conocida
y mide cuántas acierta. Sirve para comparar modelos o prompts con números.

Formato del dataset (JSON):
    {
      "reference_date": "2026-11-10",          # "hoy" para fechas relativas
      "cases": [
        {
          "name": "fecha relativa",
          "conversation": [["user", "..."], ["assistant", "..."]],
          "expected": {"commitment_date": "2026-11-13", "committed_amount": 300}
        }
      ]
    }

En `expected`, el valor "*" significa "cualquier valor no nulo": sirve para
textos libres (como `request`), donde no se puede exigir la frase exacta.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ringr_agents.agents.base import AgentData
from ringr_agents.conversation import Conversation, Message, Role
from ringr_agents.http_client import JsonScalar
from ringr_agents.models import ModelResponseError, ModelUnavailableError, ParserModelProtocol
from ringr_agents.validation import InvalidDataError

ANY_VALUE = "*"


@dataclass(frozen=True, slots=True)
class EvalCase:
    name: str
    messages: tuple[Message, ...]
    expected: Mapping[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class EvalDataset:
    reference_date: date
    cases: tuple[EvalCase, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: EvalCase
    actual: Mapping[str, JsonScalar] | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.actual is None:
            return False
        actual = self.actual
        return all(
            _matches(expected, actual.get(field)) for field, expected in self.case.expected.items()
        )


@dataclass(frozen=True, slots=True)
class EvalReport:
    results: tuple[CaseResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def accuracy(self) -> float:
        """Proporción de casos acertados, entre 0 y 1."""
        return self.passed / len(self.results) if self.results else 0.0


def load_dataset(path: Path) -> EvalDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(
        EvalCase(
            name=item["name"],
            messages=tuple(Message(Role(role), text) for role, text in item["conversation"]),
            expected=item["expected"],
        )
        for item in raw["cases"]
    )
    return EvalDataset(reference_date=date.fromisoformat(raw["reference_date"]), cases=cases)


def run_eval(
    make_parser: Callable[[Conversation], ParserModelProtocol],
    validate: Callable[[Mapping[str, Any]], AgentData],
    cases: Sequence[EvalCase],
) -> EvalReport:
    """Pasa cada caso por el parser y la misma validación que usa el agente.

    El parser lleva la conversación en su estado interno, así que cada caso
    tiene el suyo: `make_parser` lo crea a partir de la conversación del caso.
    """
    results: list[CaseResult] = []
    for case in cases:
        parser = make_parser(Conversation(messages=case.messages))
        try:
            actual = validate(parser.parse_data()).to_payload()
        except ModelUnavailableError:
            raise  # la API no respondió: no es un fallo del extractor, se para todo
        except (ModelResponseError, InvalidDataError) as error:
            results.append(CaseResult(case, actual=None, error=str(error)))
        else:
            results.append(CaseResult(case, actual=actual))
    return EvalReport(tuple(results))


def _matches(expected: JsonScalar, actual: JsonScalar) -> bool:
    if expected == ANY_VALUE:
        return actual is not None
    return expected == actual
