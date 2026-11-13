"""Estado de una conversación: su identificador y el historial de mensajes."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


class Conversation:
    """Historial de una conversación.

    El agente apunta aquí cada mensaje. Los modelos reales la reciben al crearse
    (es parte de su estado interno) y leen `messages`, una tupla (inmutable):
    pueden leer el historial pero no modificarlo por accidente.
    """

    def __init__(
        self, conversation_id: str | None = None, *, messages: Iterable[Message] = ()
    ) -> None:
        self._id = conversation_id or uuid4().hex
        self._messages: list[Message] = list(messages)

    @property
    def id(self) -> str:
        return self._id

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message(Role.USER, content))

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(Message(Role.ASSISTANT, content))
