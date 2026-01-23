FROM python:3.14.2
LABEL authors="Kamil"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app

ENV UV_NO_DEV=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN uv sync --frozen

CMD ["uv", "run", "--", "fastapi", "dev", "--host", "0.0.0.0", "main.py"]