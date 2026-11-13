"""Autenticación de las integraciones HTTP.

`Authenticator` es el contrato: cualquier objeto que sepa devolver las cabeceras
de autenticación. Hoy solo hay Bearer Token, pero mañana podría haber API keys,
OAuth, etc. sin tocar los agentes ni las acciones.
"""

from dataclasses import dataclass, field
from typing import Protocol, Self

from ringr_agents.config import Settings


class Authenticator(Protocol):
    """Contrato de cualquier método de autenticación."""

    def auth_headers(self) -> dict[str, str]:
        """Devuelve las cabeceras que autentican la petición."""
        ...


@dataclass(frozen=True, slots=True)
class BearerTokenAuth:
    """Autenticación con cabecera `Authorization: Bearer <token>`."""

    # repr=False evita que el token aparezca en logs o trazas de error.
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        # Fallar al construir es mejor que enviar peticiones sin autenticar.
        if not self.token.strip():
            raise ValueError("El token de la API no puede estar vacío.")

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(token=settings.api_token)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}
