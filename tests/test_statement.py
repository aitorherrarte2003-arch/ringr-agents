"""El enunciado, punto por punto: el proyecto usado tal y como lo describe.

Sin helpers ni dependencias inyectadas: solo los dos modelos genéricos y el
agente, con sus valores por defecto (HTTP simulado y el token de pruebas).
"""

import contextlib
import io
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from ringr_agents import ActionOutcome, AssistanceAgent, ConversationModel, DebtAgent, ParserModel

STATEMENT_TOKEN = "ringr_test_token_9f3a2c1d"
PAYMENT_DAY = (date.today() + timedelta(days=7)).isoformat()
README = Path(__file__).parent.parent / "README.md"


@pytest.fixture(autouse=True)
def statement_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin RINGR_API_TOKEN, el token es el del enunciado (aunque la máquina tenga otro)."""
    monkeypatch.delenv("RINGR_API_TOKEN", raising=False)


def test_the_generic_models_keep_everything_in_their_state() -> None:
    """Enunciado: ConversationModel "implementa el método answer_user que devuelve un
    str", ParserModel "implementa el método parse_data que devuelve un dict", y los dos
    "contienen toda la información necesaria en su estado interno"."""
    conversation_model = ConversationModel("¿Cuándo podría pagar?")
    parser_model = ParserModel({"commitment_date": None, "committed_amount": None})

    assert conversation_model.answer_user() == "¿Cuándo podría pagar?"
    assert parser_model.parse_data() == {"commitment_date": None, "committed_amount": None}


def test_debt_agent_registers_the_commitment_when_both_values_are_known() -> None:
    agent = DebtAgent(
        ConversationModel("Anotado: 150,50 € la semana que viene."),
        ParserModel({"commitment_date": PAYMENT_DAY, "committed_amount": 150.5}),
    )

    result = agent.handle_turn()

    assert result.reply == "Anotado: 150,50 € la semana que viene."
    assert result.outcome is ActionOutcome.EXECUTED
    request = result.request
    assert request is not None
    assert request.method == "POST"
    assert request.url == "https://api.ringr.debt/v1/commitment"
    assert request.headers["Authorization"] == f"Bearer {STATEMENT_TOKEN}"
    # "los datos devueltos por el ParserModel ... como headers con los mismos nombres"
    assert request.headers["commitment_date"] == PAYMENT_DAY
    assert request.headers["committed_amount"] == "150.5"
    assert request.body == {"commitment_date": PAYMENT_DAY, "committed_amount": 150.5}
    assert result.response is not None
    assert result.response.status_code == 200


@pytest.mark.parametrize(
    "parsed",
    [
        {"commitment_date": None, "committed_amount": None},
        {"commitment_date": PAYMENT_DAY, "committed_amount": None},
        {"commitment_date": None, "committed_amount": 150.5},
    ],
)
def test_debt_agent_does_nothing_until_both_values_are_known(parsed: dict[str, Any]) -> None:
    agent = DebtAgent(ConversationModel("¿Me lo puede concretar?"), ParserModel(parsed))

    result = agent.handle_turn()

    assert result.outcome is ActionOutcome.NOT_READY
    assert result.request is None


def test_assistance_agent_registers_the_request() -> None:
    agent = AssistanceAgent(
        ConversationModel("Un compañero gestionará su baja."),
        ParserModel({"request": "Darme de baja"}),
    )

    request = agent.handle_turn().request

    assert request is not None
    assert request.method == "POST"
    assert request.url == "https://api.ringr.assistance/v1/request"
    assert request.headers["Authorization"] == f"Bearer {STATEMENT_TOKEN}"
    assert request.headers["request"] == "Darme de baja"
    assert request.body == {"request": "Darme de baja"}


def test_assistance_agent_only_answers_questions_without_a_request() -> None:
    agent = AssistanceAgent(
        ConversationModel("Abrimos de 9:00 a 18:00."), ParserModel({"request": None})
    )

    assert agent.handle_turn().request is None


@pytest.mark.parametrize(
    ("agent_class", "parsed"),
    [
        (DebtAgent, {"commitment_date": PAYMENT_DAY, "committed_amount": 150.5}),
        (AssistanceAgent, {"request": "Darme de baja"}),
    ],
)
def test_no_duplicates_are_sent_during_the_conversation(
    agent_class: type[DebtAgent | AssistanceAgent], parsed: dict[str, Any]
) -> None:
    agent = agent_class(ConversationModel("Hecho."), ParserModel(parsed))

    outcomes = [agent.handle_turn().outcome for _ in range(3)]

    assert outcomes == [ActionOutcome.EXECUTED, ActionOutcome.DUPLICATE, ActionOutcome.DUPLICATE]


def test_any_object_with_the_methods_works_without_subclassing() -> None:
    """Enunciado: "no hacen falta subclases". Basta con tener el método."""

    class MyConversationModel:
        def answer_user(self) -> str:
            return "Le paso con un compañero."

    class MyParserModel:
        def parse_data(self) -> dict[str, Any]:
            return {"request": "Cambio de tarifa"}

    result = AssistanceAgent(MyConversationModel(), MyParserModel()).handle_turn()

    assert result.outcome is ActionOutcome.EXECUTED


def test_the_readme_quick_start_works_as_written() -> None:
    """El ejemplo "Pruébalo en 1 minuto" del README se ejecuta tal cual."""
    section = README.read_text(encoding="utf-8").split("### Pruébalo en 1 minuto")[1]
    section = section.split("\n#")[0]  # hasta el siguiente título
    code = "\n".join(re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL))

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code, {})  # código del propio README, no de fuera

    printed = output.getvalue()
    assert f"Bearer {STATEMENT_TOKEN}" in printed
    assert "executed" in printed
    assert "duplicate" in printed
    assert "Darme de baja" in printed
