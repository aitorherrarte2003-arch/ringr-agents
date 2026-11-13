"""Configuración del proyecto, leída del entorno.

Los secretos (como el token de la API) no se escriben en el código de los agentes:
se leen de variables de entorno. El token de pruebas del enunciado se usa solo
como valor por defecto para que el proyecto funcione sin configurar nada.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

API_TOKEN_ENV_VAR = "RINGR_API_TOKEN"

# Token de pruebas facilitado en el enunciado. En producción vendría siempre del entorno.
DEFAULT_TEST_TOKEN = "ringr_test_token_9f3a2c1d"


@dataclass(frozen=True, slots=True)
class Settings:
    """Valores de configuración que necesita la aplicación."""

    # repr=False: si alguien imprime o loguea Settings, el token no aparece.
    api_token: str = field(repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Construye la configuración a partir de las variables de entorno.

        `environ` permite pasar un diccionario en los tests en lugar de tocar
        las variables de entorno reales del proceso.
        """
        env = os.environ if environ is None else environ
        return cls(api_token=env.get(API_TOKEN_ENV_VAR, DEFAULT_TEST_TOKEN))
