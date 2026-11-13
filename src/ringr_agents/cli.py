"""Línea de comandos.

    python -m ringr_agents demo                     # conversaciones simuladas, sin API key
    python -m ringr_agents chat --agent debt        # hablar con el agente usando Claude
    python -m ringr_agents eval --agent debt        # medir el parser con el dataset

`chat` y `eval` necesitan el extra `llm` y credenciales de Anthropic
(variable ANTHROPIC_API_KEY). Las acciones HTTP siempre son simuladas: los
endpoints del enunciado son ficticios.
"""

import argparse
import io
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ringr_agents.agents import (
    AgentData,
    AssistanceAgent,
    AssistanceRequest,
    BaseAgent,
    DebtAgent,
    DebtCommitment,
)
from ringr_agents.conversation import Conversation
from ringr_agents.demo import build_scenarios
from ringr_agents.evals import EvalReport, load_dataset, run_eval
from ringr_agents.llm.profiles import ASSISTANCE_PROFILE, DEBT_PROFILE, AgentProfile
from ringr_agents.models import ModelResponseError, ModelUnavailableError
from ringr_agents.results import TurnResult

if TYPE_CHECKING:
    # Solo para el comprobador de tipos: en ejecución, `anthropic` se importa
    # únicamente cuando hace falta, así `demo` funciona sin tenerlo instalado.
    import anthropic

DEFAULT_DATASETS = Path("evals")
EXIT_WORDS = frozenset({"salir", "exit", "quit"})


@dataclass(frozen=True, slots=True)
class AgentSetup:
    """Todo lo necesario para montar un agente desde la línea de comandos."""

    agent_class: type[BaseAgent[Any]]
    profile: AgentProfile
    # Dada la fecha de "hoy", devuelve la función que valida la salida del parser.
    validator: Callable[[date], Callable[[Mapping[str, Any]], AgentData]]


AGENTS: dict[str, AgentSetup] = {
    "debt": AgentSetup(
        DebtAgent, DEBT_PROFILE, lambda today: partial(DebtCommitment.from_raw, today=today)
    ),
    "assistance": AgentSetup(
        AssistanceAgent, ASSISTANCE_PROFILE, lambda today: AssistanceRequest.from_raw
    ),
}


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], "anthropic.Anthropic"] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Punto de entrada. Devuelve el código de salida del proceso (0 = bien).

    `client_factory` e `input_fn` existen para poder testear `chat` y `eval`
    con un cliente falso y entradas simuladas.
    """
    _use_utf8_output()  # antes de parse_args: `--help` también lleva tildes
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "demo":
        return _run_demo()

    import anthropic  # import diferido: es una dependencia opcional

    factory = client_factory or anthropic.Anthropic
    try:
        client = factory()
        if not _has_credentials(client):
            print("\nFaltan las credenciales de Anthropic: define la variable ANTHROPIC_API_KEY.")
            return 1
        if args.command == "chat":
            return _run_chat(client, AGENTS[args.agent], args.model, input_fn)
        dataset = args.dataset or DEFAULT_DATASETS / f"{args.agent}.json"
        return _run_eval(client, AGENTS[args.agent], args.model, dataset)
    except (anthropic.AnthropicError, ModelUnavailableError) as error:
        print(f"\nError con la API de Anthropic: {error}")
        return 1


def _has_credentials(client: "anthropic.Anthropic") -> bool:
    """El SDK crea el cliente aunque no haya credenciales y no falla hasta la
    primera llamada (con un TypeError poco claro): mejor avisar antes."""
    return any(
        credential is not None
        for credential in (client.api_key, client.auth_token, client.credentials)
    )


def _use_utf8_output() -> None:
    """En Windows, la salida redirigida usa cp1252 y estropea tildes y "€"."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ringr_agents", description="Agentes conversacionales de Ringr."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="muestra los logs")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("demo", help="conversaciones simuladas, sin API key")

    for name, help_text in (("chat", "conversa con un agente real"), ("eval", "evalúa el parser")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--agent", choices=sorted(AGENTS), required=True)
        command.add_argument("--model", default="claude-opus-5-5")
        if name == "eval":
            command.add_argument("--dataset", type=Path, help="por defecto evals/<agent>.json")
    return parser


def _run_demo() -> int:
    for scenario in build_scenarios(today=date.today()):
        print(f"\n=== {scenario.title} ===")
        for message in scenario.user_messages:
            result = scenario.agent.handle_turn(message)
            print(format_turn(result, user_message=message), end="\n\n")
    return 0


def _run_chat(
    client: "anthropic.Anthropic",
    setup: AgentSetup,
    model: str,
    input_fn: Callable[[str], str],
) -> int:
    from ringr_agents.llm.claude import ClaudeConversationModel, ClaudeParserModel

    # La misma conversación para el agente (que la rellena) y los dos modelos
    # (que la leen): es parte de su estado interno.
    conversation = Conversation()
    agent = setup.agent_class(
        ClaudeConversationModel(
            client,
            conversation=conversation,
            instructions=setup.profile.conversation_instructions,
            model=model,
        ),
        ClaudeParserModel(
            client,
            conversation=conversation,
            instructions=setup.profile.parser_instructions,
            schema=setup.profile.parser_schema,
            model=model,
        ),
        conversation=conversation,
    )
    print(f"Chat con {setup.agent_class.__name__} ({model}). Escribe 'salir' para terminar.")
    while True:
        try:
            text = input_fn("\nTú: ").strip()
        except EOFError:  # Ctrl+Z / Ctrl+D o fin de la entrada
            break
        if text.lower() in EXIT_WORDS:
            break
        if not text:
            continue
        try:
            result = agent.handle_turn(text)
        except ModelResponseError as error:
            print(f"  (el modelo no pudo responder: {error})")
            continue
        print(format_turn(result))
    return 0


def _run_eval(client: "anthropic.Anthropic", setup: AgentSetup, model: str, path: Path) -> int:
    from ringr_agents.llm.claude import ClaudeParserModel

    try:
        data = load_dataset(path)
    except FileNotFoundError:
        print(
            f"\nNo existe el dataset {path}: ejecuta desde la carpeta del proyecto o usa --dataset."
        )
        return 1

    def make_parser(conversation: Conversation) -> ClaudeParserModel:
        return ClaudeParserModel(
            client,
            conversation=conversation,
            instructions=setup.profile.parser_instructions,
            schema=setup.profile.parser_schema,
            model=model,
            today=lambda: data.reference_date,  # fecha fija: resultados reproducibles
        )

    report = run_eval(make_parser, setup.validator(data.reference_date), data.cases)
    print(format_report(report, title=f"{path} con {model}"))
    return 0


def format_turn(result: TurnResult[Any], *, user_message: str | None = None) -> str:
    """Resumen legible de un turno: respuesta, datos y decisión."""
    lines = [f"  Usuario : {user_message}"] if user_message is not None else []
    lines += [
        f"  Agente  : {result.reply}",
        f"  Datos   : {result.data}",
        f"  Acción  : {result.outcome.value}" + (f" ({result.error})" if result.error else ""),
    ]
    if result.action_executed and result.request is not None:
        lines += [
            f"  POST    : {result.request.url}",
            f"  Headers : {result.request.redacted_headers()}",
            f"  Body    : {result.request.json_body()}",
        ]
    return "\n".join(lines)


def format_report(report: EvalReport, *, title: str) -> str:
    """Resultado de una evaluación, caso a caso y en total."""
    lines = [f"Evaluación: {title}", ""]
    for result in report.results:
        lines.append(f"[{'OK' if result.passed else 'FALLO'}] {result.case.name}")
        if not result.passed:
            lines.append(f"        esperado: {dict(result.case.expected)}")
            obtained = result.error if result.actual is None else dict(result.actual)
            lines.append(f"        obtenido: {obtained}")
    total = len(report.results)
    lines += ["", f"Aciertos: {report.passed}/{total} ({report.accuracy:.0%})"]
    return "\n".join(lines)
