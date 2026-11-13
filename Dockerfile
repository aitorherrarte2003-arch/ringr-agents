FROM python:3.12-slim

# Sin ficheros .pyc y con la salida sin búfer (los logs aparecen al momento).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
RUN pip install --no-cache-dir ".[llm,http]"

# No ejecutar como root.
RUN useradd --create-home ringr
USER ringr

ENTRYPOINT ["python", "-m", "ringr_agents"]
CMD ["demo"]
