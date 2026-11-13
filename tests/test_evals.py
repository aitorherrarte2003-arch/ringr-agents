from collections.abc import Callable, Mapping
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from ringr_agents.agents import AgentData, AssistanceRequest, DebtCommitment
from ringr_agents.conversation import Conversation, Message, Role
from ringr_agents.evals import EvalCase, EvalReport, load_dataset, run_eval
from ringr_agents.models import ModelResponseError, ModelUnavailableError, ParserModel

DATASETS = Path(__file__).parent.parent / "evals"
TODAY = date(2026, 11, 10)
VALIDATE_DEBT = partial(DebtCommitment.from_raw, today=TODAY)


def case(name: str, expected: dict[str, Any]) -> EvalCase:
    return EvalCase(name, (Message(Role.USER, "..."),), expected)


def parsers(*outputs: dict[str, Any]) -> Callable[[Conversation], ParserModel]:
    """Crea un parser simulado por caso: el primero devuelve outputs[0], etc."""
    pending = iter(outputs)
    return lambda conversation: ParserModel(next(pending))


class _FailingParser:
    def __init__(self, conversation: Conversation) -> None: ...

    def parse_data(self) -> dict[str, Any]:
        raise ModelResponseError("el modelo rechazó la petición")


def test_each_case_gets_a_parser_with_its_own_conversation() -> None:
    seen: list[tuple[Message, ...]] = []

    def make_parser(conversation: Conversation) -> ParserModel:
        seen.append(conversation.messages)
        return ParserModel({"request": None})

    first, second = (Message(Role.USER, "uno"),), (Message(Role.USER, "dos"),)
    run_eval(
        make_parser,
        AssistanceRequest.from_raw,
        [EvalCase("a", first, {}), EvalCase("b", second, {})],
    )

    assert seen == [first, second]


def test_correct_and_wrong_answers_are_counted() -> None:
    cases = [
        case("bien", {"commitment_date": "2026-11-20", "committed_amount": 150}),
        case("mal", {"commitment_date": "2026-11-20", "committed_amount": 150}),
    ]
    make_parser = parsers(
        {"commitment_date": "2026-11-20", "committed_amount": "150,00"},  # normalizado = OK
        {"commitment_date": "2026-11-21", "committed_amount": 150},
    )

    report = run_eval(make_parser, VALIDATE_DEBT, cases)

    assert [result.passed for result in report.results] == [True, False]
    assert report.passed == 1
    assert report.accuracy == 0.5


def test_any_value_accepts_any_non_null_text() -> None:
    cases = [case("pide algo", {"request": "*"}), case("no pide", {"request": "*"})]
    make_parser = parsers({"request": "Cambio de tarifa"}, {"request": None})

    report = run_eval(make_parser, AssistanceRequest.from_raw, cases)

    assert [result.passed for result in report.results] == [True, False]


def test_invalid_data_counts_as_failure_with_its_error() -> None:
    make_parser = parsers({"commitment_date": "2020-01-01", "committed_amount": 1})

    report = run_eval(make_parser, VALIDATE_DEBT, [case("pasado", {"commitment_date": None})])

    result = report.results[0]
    assert not result.passed
    assert result.actual is None
    assert result.error is not None
    assert "ya ha pasado" in result.error


def test_model_errors_count_as_failure() -> None:
    report = run_eval(_FailingParser, AssistanceRequest.from_raw, [case("x", {"request": None})])

    assert report.results[0].error == "el modelo rechazó la petición"
    assert report.accuracy == 0.0


def test_an_api_outage_stops_the_evaluation() -> None:
    # Si la API no responde, no es un fallo del extractor: contarlo como fallo
    # bajaría el porcentaje sin motivo. Mejor parar y que se repita la medición.
    class _DownParser:
        def __init__(self, conversation: Conversation) -> None: ...

        def parse_data(self) -> dict[str, Any]:
            raise ModelUnavailableError("La API de Anthropic no respondió")

    with pytest.raises(ModelUnavailableError):
        run_eval(_DownParser, AssistanceRequest.from_raw, [case("x", {"request": None})])


def test_empty_report() -> None:
    assert EvalReport(()).accuracy == 0.0


def test_load_dataset(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(
        '{"reference_date": "2026-11-10", "cases": [{"name": "n", '
        '"conversation": [["user", "hola"], ["assistant", "buenas"]], '
        '"expected": {"request": null}}]}',
        encoding="utf-8",
    )

    dataset = load_dataset(path)

    assert dataset.reference_date == TODAY
    assert dataset.cases == (
        EvalCase(
            "n", (Message(Role.USER, "hola"), Message(Role.ASSISTANT, "buenas")), {"request": None}
        ),
    )


@pytest.mark.parametrize(
    ("file_name", "make_validator"),
    [
        ("debt.json", lambda today: partial(DebtCommitment.from_raw, today=today)),
        ("assistance.json", lambda today: AssistanceRequest.from_raw),
    ],
)
def test_bundled_datasets_are_consistent(
    file_name: str,
    make_validator: Callable[[date], Callable[[Mapping[str, Any]], AgentData]],
) -> None:
    """Cada respuesta esperada del dataset debe ser válida y ya estar normalizada;
    si no, el caso sería imposible de acertar."""
    dataset = load_dataset(DATASETS / file_name)
    validate = make_validator(dataset.reference_date)

    assert len(dataset.cases) >= 8
    assert len({c.name for c in dataset.cases}) == len(dataset.cases)  # nombres únicos
    for eval_case in dataset.cases:
        assert eval_case.messages[0].role is Role.USER
        assert eval_case.messages[-1].role is Role.ASSISTANT
        assert validate(eval_case.expected).to_payload() == eval_case.expected
