"""Línea de comandos: demo, chat y eval (con cliente de Anthropic falso)."""

import io
import json
import runpy
from collections.abc import Callable, Iterator
from pathlib import Path

import anthropic
import httpx2  # la versión de httpx que usa por dentro el SDK de Anthropic
import pytest

from ringr_agents.cli import main
from tests.fakes import FakeAnthropic, text_response


def scripted_input(*lines: str) -> Callable[[str], str]:
    """Simula lo que teclea el usuario; al acabarse, se comporta como Ctrl+D."""
    pending: Iterator[str] = iter(lines)

    def read(prompt: str) -> str:
        try:
            return next(pending)
        except StopIteration:
            raise EOFError from None

    return read


def test_demo_runs_both_agents(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 0

    output = capsys.readouterr().out
    assert "=== Agente de cobros (DebtAgent) ===" in output
    assert "=== Agente de atención al cliente (AssistanceAgent) ===" in output
    assert output.count("Acción  : executed") == 2
    assert output.count("Acción  : duplicate") == 2
    assert "Bearer" not in output  # el token nunca se muestra


def test_output_streams_that_are_not_files_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.StringIO()  # p. ej. cuando otro programa captura la salida
    monkeypatch.setattr("sys.stdout", buffer)

    assert main(["demo"]) == 0
    assert "executed" in buffer.getvalue()


def test_verbose_flag_is_accepted() -> None:
    assert main(["-v", "demo"]) == 0


def test_python_dash_m_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["ringr_agents", "demo"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ringr_agents", run_name="__main__")
    assert exit_info.value.code == 0


def test_chat_conversation(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeAnthropic(
        text_response("Hola, ¿cuándo podría pagar?"),
        text_response('{"commitment_date": null, "committed_amount": null}'),
        text_response("Anotado: 150 € el 1 de enero de 2099."),
        text_response('{"commitment_date": "2099-01-01", "committed_amount": 150}'),
    )

    code = main(
        ["chat", "--agent", "debt"],
        client_factory=fake.as_client,
        input_fn=scripted_input("Hola", "   ", "150 el 1 de enero de 2099", "salir"),
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "Chat con DebtAgent (claude-opus-5-5)" in output
    assert "Acción  : not_ready" in output
    assert "Acción  : executed" in output
    assert "POST    : https://api.ringr.debt/v1/commitment" in output
    assert len(fake.messages.calls) == 4  # la línea en blanco no genera llamadas


def test_chat_ends_on_end_of_input() -> None:
    assert (
        main(
            ["chat", "--agent", "assistance"],
            client_factory=FakeAnthropic().as_client,
            input_fn=scripted_input(),
        )
        == 0
    )


def test_chat_survives_a_model_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeAnthropic(text_response("x", stop_reason="refusal"))

    code = main(
        ["chat", "--agent", "assistance"],
        client_factory=fake.as_client,
        input_fn=scripted_input("hola"),
    )

    assert code == 0
    assert "el modelo no pudo responder" in capsys.readouterr().out


def test_api_errors_are_reported(capsys: pytest.CaptureFixture[str]) -> None:
    def no_credentials() -> anthropic.Anthropic:
        raise anthropic.AnthropicError("faltan credenciales")

    assert main(["chat", "--agent", "debt"], client_factory=no_credentials) == 1
    assert "Error con la API de Anthropic: faltan credenciales" in capsys.readouterr().out


def test_missing_credentials_are_reported_before_calling_the_api(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeAnthropic(api_key=None)  # como Anthropic() sin ANTHROPIC_API_KEY

    assert main(["chat", "--agent", "debt"], client_factory=fake.as_client) == 1
    assert "Faltan las credenciales de Anthropic" in capsys.readouterr().out
    assert fake.messages.calls == []


def test_chat_survives_an_api_outage(capsys: pytest.CaptureFixture[str]) -> None:
    outage = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x.test"))
    fake = FakeAnthropic(outage)

    code = main(
        ["chat", "--agent", "debt"], client_factory=fake.as_client, input_fn=scripted_input("hola")
    )

    assert code == 0
    assert "el modelo no pudo responder" in capsys.readouterr().out


def test_eval_stops_with_a_clear_message_if_the_api_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(Path(__file__).parent.parent)  # para encontrar evals/
    outage = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x.test"))
    fake = FakeAnthropic(outage)

    code = main(["eval", "--agent", "assistance"], client_factory=fake.as_client)

    assert code == 1
    assert "Error con la API de Anthropic" in capsys.readouterr().out


def test_missing_dataset_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "no-existe.json"

    code = main(
        ["eval", "--agent", "debt", "--dataset", str(missing)],
        client_factory=FakeAnthropic().as_client,
    )

    assert code == 1
    assert "No existe el dataset" in capsys.readouterr().out


def test_eval_prints_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "mini.json"
    dataset.write_text(
        json.dumps(
            {
                "reference_date": "2026-11-10",
                "cases": [
                    {
                        "name": "acierto",
                        "conversation": [["user", "Quiero la baja"], ["assistant", "Vale"]],
                        "expected": {"request": "*"},
                    },
                    {
                        "name": "fallo",
                        "conversation": [["user", "Hola"], ["assistant", "Hola"]],
                        "expected": {"request": None},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    fake = FakeAnthropic(
        text_response('{"request": "Baja de la línea"}'),
        text_response('{"request": "Algo inventado"}'),
    )

    code = main(
        ["eval", "--agent", "assistance", "--dataset", str(dataset)],
        client_factory=fake.as_client,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "[OK] acierto" in output
    assert "[FALLO] fallo" in output
    assert "esperado: {'request': None}" in output
    assert "obtenido: {'request': 'Algo inventado'}" in output
    assert "Aciertos: 1/2 (50%)" in output
    # La fecha de referencia del dataset llega al parser.
    assert fake.messages.calls[0]["system"].endswith("Fecha de hoy: 2026-11-10.")


def test_eval_uses_the_bundled_dataset_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(Path(__file__).parent.parent)
    answers = [text_response('{"request": null}') for _ in range(20)]

    main(["eval", "--agent", "assistance"], client_factory=FakeAnthropic(*answers).as_client)

    assert "Evaluación: evals" in capsys.readouterr().out


def test_eval_shows_errors_of_failed_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "d.json"
    dataset.write_text(
        '{"reference_date": "2026-11-10", "cases": [{"name": "roto", '
        '"conversation": [["user", "x"], ["assistant", "y"]], "expected": {"request": null}}]}',
        encoding="utf-8",
    )
    fake = FakeAnthropic(text_response("no es json"))

    main(
        ["eval", "--agent", "assistance", "--dataset", str(dataset)], client_factory=fake.as_client
    )

    assert "obtenido: El parser no devolvió JSON válido" in capsys.readouterr().out
