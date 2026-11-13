# Ringr Agents

Agentes conversacionales que, en cada turno, **responden al usuario**, **extraen datos
estructurados** de la conversación, los **validan** y, si se cumplen las reglas de negocio,
**ejecutan una acción HTTP**, sin enviar nunca duplicados.

| Agente | Extrae | Actúa cuando… | Endpoint |
|---|---|---|---|
| `DebtAgent` | `commitment_date`, `committed_amount` | hay fecha **y** importe | `POST https://api.ringr.debt/v1/commitment` |
| `AssistanceAgent` | `request` | hay una solicitud | `POST https://api.ringr.assistance/v1/request` |

El núcleo pedido en el enunciado no tiene dependencias externas (solo la biblioteca estándar).
Encima de él, como **extras opcionales que no tocan el núcleo**, hay un adaptador a un LLM real
(Claude), un cliente HTTP real con reintentos, un chat por terminal y evaluaciones del extractor.

**¿Prisa?** [Pruébalo en 1 minuto](#pruébalo-en-1-minuto), con los mismos nombres que el
enunciado: `DebtAgent(ConversationModel(...), ParserModel(...)).handle_turn()`.

---

## Índice

1. [Puesta en marcha](#1-puesta-en-marcha)
2. [Cómo funciona un turno](#2-cómo-funciona-un-turno)
3. [Arquitectura](#3-arquitectura)
4. [Decisiones técnicas](#4-decisiones-técnicas)
5. [Supuestos ante las ambigüedades del enunciado](#5-supuestos-ante-las-ambigüedades-del-enunciado)
6. [Más allá del enunciado](#6-más-allá-del-enunciado)
7. [Cómo añadir un agente nuevo](#7-cómo-añadir-un-agente-nuevo)
8. [Calidad](#8-calidad)
9. [Qué cambiaría para llevarlo a producción](#9-qué-cambiaría-para-llevarlo-a-producción)

---

## 1. Puesta en marcha

Requiere **Python 3.12+**. Los comandos se ejecutan desde la carpeta del proyecto.

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
```

### Pruébalo en 1 minuto

Tal y como lo describe el enunciado: un `ConversationModel` y un `ParserModel` genéricos que
llevan en su estado lo que van a responder (sin subclases), un agente y `handle_turn()`. El
cliente HTTP simulado y el Bearer token del enunciado vienen por defecto. Basta con pegarlo en
una consola de Python (`python`) con el entorno activado.

```python
from datetime import date, timedelta

from ringr_agents import AssistanceAgent, ConversationModel, DebtAgent, ParserModel

payment_day = (date.today() + timedelta(days=7)).isoformat()  # una fecha pasada se rechaza

agent = DebtAgent(
    ConversationModel("Perfecto, queda registrado su compromiso de pago."),
    ParserModel({"commitment_date": payment_day, "committed_amount": 150.5}),
)

result = agent.handle_turn()
print(result.outcome)
print(result.request.method, result.request.url)
print(dict(result.request.headers))
print(result.request.json_body())
print(agent.handle_turn().outcome)  # los mismos datos no se envían dos veces
```

```text
executed
POST https://api.ringr.debt/v1/commitment
{'commitment_date': '2026-11-19', 'committed_amount': '150.5', 'Content-Type': 'application/json',
 'Idempotency-Key': '9be0c4…', 'Authorization': 'Bearer ringr_test_token_9f3a2c1d'}
{"commitment_date": "2026-11-19", "committed_amount": 150.5}
duplicate
```

Con un guion, un turno por elemento (al acabarse, cada modelo repite el último):

```python
agent = DebtAgent(
    ConversationModel(["¿Cuándo podría pagar?", "¿Qué importe?", "Anotado, gracias."]),
    ParserModel(
        [
            {"commitment_date": None, "committed_amount": None},
            {"commitment_date": payment_day, "committed_amount": None},
            {"commitment_date": payment_day, "committed_amount": 150.5},
        ]
    ),
)
for _ in range(4):
    result = agent.handle_turn()
    print(f"{result.outcome:<10} {result.reply}")

assistance = AssistanceAgent(
    ConversationModel("Un compañero gestionará su baja."),
    ParserModel({"request": "Darme de baja"}),
)
request = assistance.handle_turn().request
print(request.url, "·", request.headers["request"])
```

```text
not_ready  ¿Cuándo podría pagar?
not_ready  ¿Qué importe?
executed   Anotado, gracias.
duplicate  Anotado, gracias.
https://api.ringr.assistance/v1/request · Darme de baja
```

Vale cualquier objeto con `answer_user()` y `parse_data()`, sin heredar de nada. Para ver todas
las peticiones de una conversación, pásale tu propio cliente:
`DebtAgent(..., http_client=SimulatedHttpClient())` y luego `client.sent_requests`. Este ejemplo
se ejecuta en los tests (`tests/test_statement.py`, que recorre el enunciado punto por punto),
así que no puede quedarse desfasado.

### Comandos

| Comando | Qué hace |
|---|---|
| `python -m ringr_agents demo` | Una conversación simulada con cada agente. **No necesita API key.** |
| `pytest --cov` | Tests + cobertura |
| `mypy` | Comprobación de tipos en modo estricto |
| `ruff check .` y `ruff format --check .` | Lint y formato |
| `python -m ringr_agents chat --agent debt` | Hablar con el agente usando Claude (necesita `ANTHROPIC_API_KEY`) |
| `python -m ringr_agents eval --agent debt` | Medir el extractor con el dataset `evals/debt.json` (necesita `ANTHROPIC_API_KEY`) |
| `docker build -t ringr-agents .` y luego `docker run --rm ringr-agents` | La demo dentro de un contenedor |

Configuración por variables de entorno:

| Variable | Uso | Por defecto |
|---|---|---|
| `RINGR_API_TOKEN` | Bearer token de las integraciones | `ringr_test_token_9f3a2c1d` (el del enunciado) |
| `ANTHROPIC_API_KEY` | Solo para `chat` y `eval` | — |

---

## 2. Cómo funciona un turno

Cada agente expone `handle_turn() -> TurnResult`. Los modelos no reciben argumentos: lo que
necesitan está en su estado. El mensaje del usuario es opcional (`handle_turn("Hola")`): si se
pasa, el agente lo apunta en la conversación, que es de donde lo leen los modelos reales (así
funciona el chat con Claude).

```mermaid
flowchart TD
    A["handle_turn()"] --> B["1 · ConversationModel.answer_user()<br/>(respuesta al usuario)"]
    B --> C["2 · ParserModel.parse_data()<br/>(dict con los datos)"]
    C --> D{"3 · ¿datos válidos?"}
    D -- no --> X["INVALID_DATA"]
    D -- sí --> E{"4 · should_act<br/>¿regla de negocio?"}
    E -- no --> Y["NOT_READY"]
    E -- sí --> F{"5 · ¿son los últimos datos<br/>enviados (o enviándose)?"}
    F -- sí --> Z["DUPLICATE"]
    F -- no --> G["HttpClient.send"]
    G -- 2xx --> H["EXECUTED<br/>(se confirma en el registro)"]
    G -- error --> I["FAILED<br/>(se reintentará en el siguiente turno)"]
```

`TurnResult` cuenta todo lo que ha pasado: la respuesta, los datos validados, el resultado
(`outcome`), la petición construida, la respuesta del servidor y el error si lo hubo.

```text
$ python -m ringr_agents demo
=== Agente de cobros (DebtAgent) ===
  Usuario : Unos 150,50 €
  Agente  : Queda registrado: 150,50 € ese día. ¡Gracias!
  Datos   : DebtCommitment(commitment_date='2026-11-22', committed_amount=150.5)
  Acción  : executed
  POST    : https://api.ringr.debt/v1/commitment
  Headers : {'commitment_date': '2026-11-22', 'committed_amount': '150.5',
             'Content-Type': 'application/json', 'Idempotency-Key': '72a1f2d1…', 'Authorization': '***'}
  Body    : {"commitment_date": "2026-11-22", "committed_amount": 150.5}

  Usuario : ¿Seguro que ha quedado apuntado?
  Agente  : Su compromiso ya está registrado, no se preocupe.
  Acción  : duplicate
```

---

## 3. Arquitectura

```mermaid
flowchart LR
    subgraph Agente
        BA["BaseAgent<br/>(ciclo del turno)"] --> DA[DebtAgent]
        BA --> AA[AssistanceAgent]
    end
    BA -. usa .-> CM["ConversationModelProtocol<br/>«contrato»"]
    BA -. usa .-> PM["ParserModelProtocol<br/>«contrato»"]
    BA -. usa .-> HC["HttpClient<br/>«contrato»"]
    BA -. usa .-> AU["Authenticator<br/>«contrato»"]
    BA -. usa .-> LE["ActionLedger<br/>«contrato»"]
    CM --- S1["ConversationModel (simulado)"] & C1["ClaudeConversationModel (real)"]
    PM --- S2["ParserModel (simulado)"] & C2["ClaudeParserModel (real)"]
    HC --- S3["SimulatedHttpClient"] & H3["HttpxClient (real)"]
```

Los agentes dependen solo de **contratos**. Las implementaciones simuladas y las reales son
intercambiables sin cambiar ni una línea de los agentes; los tests lo demuestran
(`test_agent_works_end_to_end_with_the_claude_adapters`, `test_plugs_into_an_agent_without_changing_it`).

```text
src/ringr_agents/
├── agents/
│   ├── base.py          BaseAgent: el ciclo del turno, igual para todos (template method)
│   ├── debt.py          DebtAgent + DebtCommitment (datos validados)
│   └── assistance.py    AssistanceAgent + AssistanceRequest
├── models.py            ConversationModel / ParserModel genéricos (simulados) y sus contratos
├── validation.py        Validación y normalización: fechas, importes, textos
├── actions.py           HttpAction: endpoint + construcción de la petición (URL, headers, body)
├── idempotency.py       Registro anti-duplicados (en memoria y en SQLite) e Idempotency-Key
├── http_client.py       HttpRequest / HttpResponse / contrato HttpClient + cliente simulado
├── auth.py, config.py   Bearer token leído del entorno
├── conversation.py      Historial de la conversación
├── results.py           TurnResult y ActionOutcome
│   ─────────────── extras opcionales (no los usa el núcleo) ───────────────
├── llm/claude.py        ConversationModel y ParserModel reales con Claude
├── llm/profiles.py      Instrucciones y esquema JSON de cada agente para el LLM
├── httpx_client.py      HttpClient real con reintentos y backoff exponencial
├── evals.py             Evaluación del extractor contra conversaciones etiquetadas
├── demo.py, cli.py      Demo y línea de comandos (python -m ringr_agents)
evals/                   Datasets de evaluación (cobros y atención al cliente)
tests/                   257 tests: el enunciado punto por punto, unitarios, conversaciones e integración
```

---

## 4. Decisiones técnicas

### 4.1 Template method en `BaseAgent`: el ciclo es común y cada agente solo dice lo que cambia

Los pasos del turno (responder → extraer → validar → decidir → ejecutar sin duplicados) son
**idénticos** para cualquier agente. Por eso viven una sola vez en `BaseAgent.handle_turn`, y cada
agente concreto solo define tres cosas (y, si lo necesita, una cuarta opcional: `dedup_payload`,
ver 4.3):

```python
class DebtAgent(BaseAgent[DebtCommitment]):
    action = HttpAction(name="debt.commitment", url="https://api.ringr.debt/v1/commitment")

    def validate_data(self, raw):
        return DebtCommitment.from_raw(raw, today=self._today())

    def should_act(self, data):
        return data.commitment_date is not None and data.committed_amount is not None
```

Así, un agente nuevo no puede olvidarse de la deduplicación ni del manejo de errores, porque no
tiene que escribirlos. `action`, `validate_data` y `should_act` son **abstractos**: si un agente
no los define, Python no deja ni crearlo (hay un test para eso).

### 4.2 Inyección de dependencias y contratos (`Protocol`)

Los modelos, el cliente HTTP, la autenticación, el registro de acciones y hasta el reloj
(`today`) se **pasan desde fuera** en el constructor en lugar de crearse dentro del agente.
Esto es lo que permite:

- testear sin red, sin LLM y con "hoy" fijo;
- cambiar el modelo simulado por Claude, o el cliente simulado por uno real, sin tocar el agente.

Solo los dos modelos son obligatorios (y van primero, sin nombre). El resto tiene por defecto lo
que pide el enunciado: el cliente HTTP simulado (200 OK), el Bearer token de pruebas, un registro
anti-duplicados en memoria, una conversación nueva y la fecha de hoy.

Los contratos son `typing.Protocol` (tipado estructural): `ConversationModelProtocol` exige
`answer_user() -> str` y `ParserModelProtocol` exige `parse_data() -> dict`. Cualquier clase con
ese método los cumple sin heredar de nada, y mypy lo comprueba. Así los nombres del enunciado,
`ConversationModel` y `ParserModel`, quedan para los objetos genéricos que pide. Para `BaseAgent`
uso `ABC` porque ahí sí hay código compartido que heredar.

### 4.3 Sin duplicados: registro por conversación y acción + `Idempotency-Key`

El `ParserModel` lee **toda** la conversación en cada turno, así que, una vez el cliente se
compromete, el parser devolverá los mismos datos en todos los turnos siguientes. La regla es:
**se envía cuando los datos son distintos de los últimos enviados con éxito** en esa
conversación y para esa acción.

```text
150 € el 20 → executed · "¿seguro?" → duplicate · "mejor 200 €" → executed · "no, 150 €" → executed
```

1. Cada pareja conversación + acción es un **ámbito**. Un `ActionLedger` guarda por ámbito la
   **huella** (SHA-256, con las claves ordenadas) de los últimos datos enviados con éxito y cuántos
   envíos lleva.
2. Antes de enviar, el agente **reserva** el ámbito en un solo paso atómico: si los datos son
   los últimos enviados, o ya hay un envío en curso, la respuesta es `DUPLICATE` y no se envía
   nada. Así dos turnos simultáneos de la misma llamada no pueden enviar los dos.
3. Si el servidor responde **2xx**, la reserva se **confirma**. Si falla (o salta un error
   inesperado), se **libera** y, si hay otro turno, lo reintenta.
4. Cada envío lleva una **`Idempotency-Key`** propia: un reintento repite la clave **y el
   contenido exacto** del primer intento (el servidor puede descartar el duplicado, y uno que siga
   el estándar rechazaría una clave repetida con otro contenido), y un envío nuevo lleva otra
   clave, aunque vuelva a un valor anterior (si no, el servidor tomaría "no, 150 €" por un
   reintento del primer envío y se quedaría con 200 €).

Decisiones a propósito:

- **El ámbito incluye el id de la conversación.** Dos clientes distintos que se comprometen a
  pagar lo mismo el mismo día no son un duplicado.
- **Volver a un valor anterior se envía otra vez**, porque la API debe quedarse con el último
  compromiso del cliente.
- **`AssistanceAgent` registra una sola solicitud por conversación.** El texto de `request` lo
  redacta el LLM y cambia de un turno a otro ("Cambio a fibra 1 Gb" / "Cambiar a la fibra de
  1 Gb"); si contara para los duplicados, la misma petición se enviaría varias veces. Para eso
  existe `dedup_payload()`: cada agente decide qué datos cuentan (por defecto, todos; en
  `AssistanceAgent`, ninguno). El coste asumido: si el cliente corrige su petición más tarde en
  la misma llamada, la corrección no se envía.
- **Dos registros con el mismo contrato**: `InMemoryActionLedger` (por defecto) y
  `SqliteActionLedger`, que guarda el registro en un fichero: sobrevive a un reinicio del servicio
  y lo comparten varios procesos de la misma máquina. Si un proceso muere a mitad de un envío, su
  reserva caduca (120 s por defecto) para que la acción no se quede sin enviar. Cada reserva
  lleva un token: solo quien la tiene puede confirmarla o liberarla, así que un envío que llega
  tarde nunca pisa la reserva que otro turno recuperó. Los mismos tests del contrato se ejecutan
  contra los dos.
- **Si el registro falla** (por ejemplo, SQLite bloqueado), el turno no se rompe: antes de enviar,
  no se envía nada (`FAILED`); después de enviar, el turno termina con normalidad, porque el
  envío no tiene vuelta atrás. Queda un aviso `ledger_error` en el log.

```python
from ringr_agents.conversation import Conversation
from ringr_agents.idempotency import SqliteActionLedger

agent = DebtAgent(
    conversation_model,  # los dos modelos, como siempre
    parser_model,
    ledger=SqliteActionLedger("acciones.db"),
    conversation=Conversation("llamada-42"),  # el id de la llamada, para retomarla tras un reinicio
)
```

### 4.4 Validación estricta y explícita

Lo que devuelve un LLM no es de fiar, así que cada campo pasa por una función pura de
`validation.py` que devuelve el valor normalizado, `None` si el dato aún no existe, o lanza
`InvalidDataError` si el dato existe pero es incorrecto.

| Campo | Se acepta | Se rechaza |
|---|---|---|
| `commitment_date` | `"2026-11-20"`; espacios alrededor | formatos distintos de `yyyy-mm-dd` (`20261120`, `2026/11/20`), fechas imposibles (`2026-02-30`), **fechas pasadas** |
| `committed_amount` | `150`, `150.5`, `"150,50"` → redondeo a céntimos (half-up) | `≤ 0`, más de mil millones, booleanos (`True` es un `int` en Python), `NaN`, infinito, **formatos ambiguos** como `"1.500"` (¿1,5 € o 1.500 €?) |
| `request` | texto; se quitan los espacios sobrantes | tipos que no son texto; vacío o solo espacios cuenta como `None` |

No uso Pydantic a propósito: son tres reglas, las funciones se leen de arriba abajo, el núcleo
se queda sin dependencias y cada regla tiene su test.

### 4.5 Los datos van también en las cabeceras, codificados solo si hace falta

El enunciado pide enviar los datos parseados también como cabeceras, con el mismo nombre. Pero
una cabecera HTTP solo admite ASCII visible y espacios, y una solicitud del cliente puede traer
tildes ("información") o **saltos de línea**, con los que se podrían inyectar cabeceras falsas
(`"hola\r\nX-Admin: true"`). Por eso esos caracteres (y el propio `%`) van **percent-encoded**
(RFC 3986); todo lo demás va tal cual:

```text
request: Darme de baja                 ← cabecera igual que el dato
request: Reclamaci%C3%B3n de factura   ← con tildes: se codifica lo justo
{"request": "Reclamación de factura"}  ← el body lleva siempre el valor original
```

El receptor recupera siempre el original con `unquote`, también cuando no había nada que
codificar. Fechas e importes quedan igual (`2026-11-20`, `150.5`). Las cabeceras del sistema
(`Authorization`, `Content-Type`, `Idempotency-Key`) se escriben al final y un dato nunca puede
usar su nombre, así que nada puede sobrescribir la autenticación.

### 4.6 Qué pasa cuando algo falla

| Situación | Resultado | ¿El usuario recibe respuesta? |
|---|---|---|
| Faltan datos para actuar | `NOT_READY` | Sí |
| El parser devuelve datos incorrectos o algo que no es un dict | `INVALID_DATA` + motivo | Sí |
| El LLM extractor falla (rechazo, JSON roto, respuesta cortada, API caída o saturada) | `INVALID_DATA` + motivo | Sí |
| El servidor responde 4xx/5xx | `FAILED` + código; se reintenta en el siguiente turno | Sí |
| Error de red | `FAILED` + motivo; se reintenta en el siguiente turno | Sí |
| Falla el modelo conversacional | Se propaga `ModelResponseError`: no hay respuesta que dar (el chat lo avisa y sigue) | — |
| El registro anti-duplicados falla (p. ej. SQLite bloqueado) | Antes de enviar: `FAILED`, no se envía nada. Después: el turno termina normal (aviso en el log) | Sí |
| API key de Anthropic ausente o incorrecta | El chat y el eval se paran con un mensaje claro: no se arregla esperando | — |
| La API de Anthropic se cae durante `eval` | La evaluación se para con un mensaje: contarlo como fallo del extractor falsearía el porcentaje | — |

El agente nunca registra una acción a medias y nunca deja al usuario sin respuesta por un fallo
de extracción o de integración. Un fallo se reintenta **si hay otro turno**: si la llamada termina
justo después, el envío queda sin hacer (en producción iría a una cola de reintentos, ver 9).

### 4.7 Seguridad y privacidad

- El token se lee de `RINGR_API_TOKEN`. El del enunciado es solo el valor por defecto.
- El token **nunca aparece** al imprimir objetos: `repr` de `BearerTokenAuth`, `Settings` y
  `HttpRequest` lo ocultan (`'Authorization': '***'`).
- Los logs solo contienen agente, conversación, turno, resultado y, si algo falla, el tipo de
  fallo (`invalid_data field=commitment_date`, `http_status=503`, `ledger_error`). **No hay
  importes, fechas ni textos del cliente**, ni siquiera cuando un dato se rechaza, porque en
  cobros de deudas eso es información sensible. Hay tests de los tres casos (éxito, dato rechazado, envío fallido).

### 4.8 Inmutabilidad y resultados explícitos

`HttpRequest`, `TurnResult`, `Message` y los datos validados son `dataclass(frozen=True)`.
Además, las cabeceras y el body de la petición se envuelven en una vista de solo lectura, de modo
que nadie puede modificar una petición después de construirla. `handle_turn` no imprime nada:
devuelve un `TurnResult` que se puede testear, loguear o mostrar como se quiera.

---

## 5. Supuestos ante las ambigüedades del enunciado

1. **"Los modelos genéricos contienen toda la información en su estado interno"**: por eso
   `answer_user()` y `parse_data()` no reciben argumentos y `handle_turn()` tampoco los necesita.
   Los simulados guardan un guion (uno o varios valores; al acabarse repiten el último, como un
   parser que vuelve a leer una conversación que ya no cambia). Los de Claude guardan la
   conversación, que comparten con el agente.
2. **"Headers adicionales con los datos del ParserModel"**: se envían solo los campos
   declarados por el agente (lista cerrada), no cualquier clave que devuelva el parser. Así un
   parser no puede colar cabeceras arbitrarias. Los valores `None` no se envían como cabecera.
3. **El body** contiene los datos validados y normalizados (los mismos que las cabeceras, pero
   sin codificar).
4. **Fechas pasadas**: un compromiso de pago con fecha anterior a hoy se considera inválido
   (el error lo dice: `la fecha 2024-05-20 ya ha pasado`). Hoy sí es válido.
5. **Un compromiso que cambia** dentro de la misma conversación se registra de nuevo, también si
   vuelve a un valor anterior (ver 4.3).
6. **Una solicitud de atención por conversación** (ver 4.3): las redacciones distintas de la
   misma petición no son solicitudes nuevas.
7. **Un agente atiende una conversación.** El registro anti-duplicados puede compartirse entre
   agentes porque cada ámbito incluye el id de la conversación.
8. **Respuesta 2xx = éxito.** El enunciado dice que una petición correcta devuelve 200; también se
   aceptan 201, 204, etc. Cualquier otra respuesta es un fallo que se reintenta en el siguiente
   turno: sin el contrato real de la API no sé qué 4xx son definitivos (ver 9).
9. Los modelos reales leen el **historial completo** de la conversación (una tupla inmutable).
   El parser lo lee *después* de que el agente apunte su respuesta, porque esa respuesta suele
   contener la confirmación ("anoto 150 € para el día 20").
10. **"Hoy"** es la fecha de la máquina que ejecuta el agente. Es inyectable (`today=`), así que en
    producción se fijaría a la zona horaria del cliente.

---

## 6. Más allá del enunciado

Todo esto es opcional y está fuera del núcleo: si se borra, los agentes y sus tests siguen
funcionando igual.

### 6.1 LLM real con Claude (`llm/`)

`ClaudeConversationModel` y `ClaudeParserModel` cumplen los mismos contratos que los simulados.
Su estado interno es el cliente de Anthropic, las instrucciones, el esquema y la conversación,
la misma que el agente va rellenando:

```python
conversation = Conversation()
agent = DebtAgent(
    ClaudeConversationModel(client, conversation=conversation, instructions=...),
    ClaudeParserModel(client, conversation=conversation, instructions=..., schema=...),
    conversation=conversation,
)
agent.handle_turn("El viernes le pago 300 €")
```

- **Salida estructurada**: el parser pide a la API un JSON que cumpla un esquema
  (`output_config.format`). La API garantiza un JSON válido y la validación del agente garantiza
  que los valores tienen sentido.
- **El esquema y los datos no pueden desincronizarse**: un test comprueba que los campos del
  esquema JSON coinciden con los de `DebtCommitment` / `AssistanceRequest`.
- **Fecha de hoy en el prompt del parser**, para resolver "este viernes" o "dentro de una
  semana".
- **Rechazos y respuestas cortadas** se convierten en `ModelResponseError`, que el agente
  gestiona como `INVALID_DATA` sin romper el turno. Además se activan los *fallbacks* de la API:
  si el modelo rechaza una petición, la propia API la reintenta con otro modelo.
- Esfuerzo de razonamiento **bajo para conversar** (voz: prima la latencia) y **medio para
  extraer** (resolver fechas relativas pide más cuidado).
- La API espera que la conversación empiece y termine con el usuario. Si el agente habla primero
  (`handle_turn()` sin mensaje, como en una llamada de cobros) o dos veces seguidas, se añade una
  nota breve ("(Empieza la llamada.)", "(El cliente no dice nada.)").

```bash
pip install -e ".[llm]"
$env:ANTHROPIC_API_KEY = "..."       # PowerShell · cmd: set ANTHROPIC_API_KEY=... · Linux/macOS: export ANTHROPIC_API_KEY=...
python -m ringr_agents chat --agent debt
```

Los fallos pasajeros de la API (sin conexión, 429, 5xx o saturación), que el SDK ya reintenta por
su cuenta, se convierten en `ModelResponseError`: si falla el parser, el turno sigue
(`INVALID_DATA`), y el chat avisa y sigue en vez de cerrarse. Los errores de configuración (API
key incorrecta) no se ocultan.

### 6.2 Evaluaciones del extractor (`evals/`)

Que compile no dice si el agente entiende a clientes reales. Los datasets contienen
conversaciones con la respuesta correcta ya conocida: 11 de cobros (fechas relativas, cambios de
opinión, importes en letras, propuestas del agente que el cliente rechaza…) y 8 de atención al
cliente (dudas resueltas frente a solicitudes reales). `python -m ringr_agents eval --agent debt`
mide el porcentaje de aciertos y enseña cada fallo con lo esperado y lo obtenido. Sirve para
comparar modelos o cambios de prompt con números en lugar de impresiones.

Las respuestas esperadas pasan por **la misma validación que usa el agente**, y un test comprueba
que todas son válidas y están normalizadas (si no, serían casos imposibles de acertar). Por
defecto lee `evals/<agente>.json` desde la carpeta actual; desde otra carpeta, `--dataset` indica
el fichero.

### 6.3 Cliente HTTP real (`httpx_client.py`)

`HttpxClient` cumple el contrato `HttpClient`: timeouts y **reintentos con backoff exponencial**
para errores de red, 429 y 5xx (un 4xx no se reintenta). Reintentar es seguro si el servidor
respeta la `Idempotency-Key`, como es la práctica estándar. Está probado contra un transporte
simulado de httpx, incluido un agente completo enviando por él.

### 6.4 Línea de comandos, CI y Docker

- `python -m ringr_agents {demo,chat,eval}` (o `ringr-agents` tras instalar).
- GitHub Actions (`.github/workflows/ci.yml`): ruff, mypy estricto, tests con cobertura mínima del
  95 % y la demo, en Python 3.12 y 3.13.
- `Dockerfile` con un usuario sin privilegios; por defecto ejecuta la demo.

---

## 7. Cómo añadir un agente nuevo

Ejemplo: un agente que agenda una rellamada cuando el cliente da una fecha.

**1. Los datos validados**: qué se extrae y cómo se valida, reutilizando `validation.py`.

```python
@dataclass(frozen=True)
class CallbackData:
    callback_date: str | None

    def to_payload(self) -> Mapping[str, JsonScalar]:
        return {"callback_date": self.callback_date}
```

**2. El agente**: endpoint, validación y regla de negocio.

```python
class CallbackAgent(BaseAgent[CallbackData]):
    action = HttpAction(name="callback.schedule", url="https://api.ringr.example/v1/callback")

    def validate_data(self, raw):
        return CallbackData(
            optional_iso_date(
                raw.get("callback_date"), field="callback_date", not_before=self._today()
            )
        )

    def should_act(self, data):
        return data.callback_date is not None
```

Y se usa igual que los demás: `CallbackAgent(ConversationModel(...), ParserModel(...))`.

**3. (Opcional) Qué cuenta para los duplicados**: si no todos los datos deben contar, el agente
redefine `dedup_payload()` (como hace `AssistanceAgent`).

**4. (Opcional) Su perfil para el LLM** en `llm/profiles.py`: instrucciones y esquema JSON.

Respuesta al usuario, deduplicación, autenticación, cabeceras, errores y logs vienen incluidos.
Este agente existe de verdad en `tests/test_base_agent.py` y tiene sus tests.

---

## 8. Calidad

- **257 tests** con **100 % de cobertura de líneas y ramas**: unitarios por pieza, conversaciones
  completas de varios turnos (incluidos duplicados, cambios de opinión, reintentos tras 500 y
  errores de red), integración con los adaptadores reales usando clientes falsos y tests de la CLI.
- `tests/test_statement.py` recorre **el enunciado punto por punto**, usando el proyecto
  exactamente como lo describe (solo `ConversationModel`, `ParserModel`, el agente y
  `handle_turn()`), y ejecuta el ejemplo de [Pruébalo en 1 minuto](#pruébalo-en-1-minuto).
- Los tests del registro anti-duplicados se ejecutan contra **las dos implementaciones**, con
  hilos simultáneos. El del cerrojo está comprobado a la inversa: sin el cerrojo, falla.
- Pasan en **Python 3.12 y 3.13** (los dos que prueba la CI).
- **mypy en modo estricto** sin un solo error, también en los tests.
- **ruff** (lint + formato) con reglas de bugs, simplificación, nombres y pytest.
- Los tests nunca usan la red: los LLM y el HTTP se simulan con objetos que cumplen los mismos
  contratos.

---

## 9. Qué cambiaría para llevarlo a producción

- **Registro anti-duplicados entre máquinas**: `SqliteActionLedger` cubre los reinicios y varios
  procesos de una máquina. Con réplicas en varias máquinas, el mismo contrato (`reserve`,
  `confirm`, `release`) se implementaría sobre PostgreSQL o Redis, con la misma reserva atómica.
- **Cola de reintentos (outbox)**: hoy un envío fallido se reintenta en el siguiente turno; si la
  llamada termina justo después, se pierde. Y el contenido que se reenvía en un reintento se
  guarda en la memoria del agente, así que tras un reinicio se pierde. En producción, una cola
  guardaría el envío completo y lo reintentaría aunque la conversación haya acabado.
- **Contrato de errores de la API**: distinguir los 4xx definitivos (no reintentar) de un 409
  "ya registrado" (darlo por confirmado).
- **Avisar al modelo conversacional** cuando un dato es inválido, para que vuelva a preguntarlo:
  hoy la respuesta se genera antes de extraer los datos (es el orden del enunciado) y no se entera.
- **Zona horaria del cliente** para "hoy" (un cliente en Canarias va una hora por detrás).
- **Async** (`async def handle_turn`) si el servicio atiende muchas llamadas a la vez.
- **URLs por entorno** (staging/producción) en la configuración, en lugar de fijas en cada agente.
- **Secretos** desde un gestor (AWS Secrets Manager, Vault…) en lugar de variables de entorno.
- **Observabilidad**: logs en JSON y trazas (OpenTelemetry) por conversación.
- **Cabeceras**: si una solicitud pudiera ser muy larga, conviene limitar su tamaño, porque muchos
  servidores rechazan cabeceras de más de 8 KB. Además, nginx descarta por defecto las cabeceras
  con guion bajo (`underscores_in_headers off`), así que habría que acordar el nombre con el
  equipo que recibe la petición.
- **Evaluaciones en CI** con un dataset mayor y un umbral mínimo de aciertos antes de desplegar
  un cambio de prompt o de modelo.
