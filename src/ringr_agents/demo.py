"""Conversaciones de demostración con modelos simulados (no necesitan API key).

Solo construye los escenarios; quien los ejecuta y muestra es la CLI
(`python -m ringr_agents demo`).
"""

from dataclasses import dataclass
from datetime import date, timedelta

from ringr_agents.agents import AssistanceAgent, DebtAgent
from ringr_agents.models import ConversationModel, ParserModel


@dataclass(frozen=True, slots=True)
class DemoScenario:
    title: str
    agent: DebtAgent | AssistanceAgent
    # Un mensaje del usuario por turno. None: el turno no lleva mensaje (el
    # agente habla primero, como en una llamada de cobros).
    user_messages: tuple[str | None, ...]


def build_scenarios(today: date) -> tuple[DemoScenario, ...]:
    payment_day = (today + timedelta(days=10)).isoformat()

    debt = DemoScenario(
        title="Agente de cobros (DebtAgent)",
        agent=DebtAgent(
            ConversationModel(
                [
                    "Hola, le llamo de Ringr por un recibo pendiente de 300 €. "
                    "¿Cuándo podría pagarlo?",
                    "Perfecto. ¿Qué importe podría abonar ese día?",
                    "Queda registrado: 150,50 € ese día. ¡Gracias!",
                    "Su compromiso ya está registrado, no se preocupe.",
                ]
            ),
            ParserModel(
                [
                    {"commitment_date": None, "committed_amount": None},
                    {"commitment_date": payment_day, "committed_amount": None},
                    {"commitment_date": payment_day, "committed_amount": "150,50"},
                    {"commitment_date": payment_day, "committed_amount": "150,50"},
                ]
            ),
            today=lambda: today,
        ),
        user_messages=(
            None,
            f"Podría pagar el {payment_day}",
            "Unos 150,50 €",
            "¿Seguro que ha quedado apuntado?",
        ),
    )

    assistance = DemoScenario(
        title="Agente de atención al cliente (AssistanceAgent)",
        agent=AssistanceAgent(
            ConversationModel(
                [
                    "Nuestro horario es de 9:00 a 18:00. ¿Algo más?",
                    "Entendido, paso su solicitud a un compañero.",
                    "Ya está registrada, le llamarán en 24 h.",
                ]
            ),
            ParserModel(
                [
                    {"request": None},
                    {"request": "Cambiar la tarifa a la de fibra 1 Gb"},
                    {"request": "Cambiar la tarifa a la de fibra 1 Gb"},
                ]
            ),
            today=lambda: today,
        ),
        user_messages=(
            "¿Qué horario tenéis?",
            "Quiero cambiarme a la fibra de 1 Gb",
            "Vale, gracias",
        ),
    )

    return (debt, assistance)
