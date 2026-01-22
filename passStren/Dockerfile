FROM python:3.14.2
LABEL authors="Kamil"

COPY . /app

ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --frozen

CMD["uv", "run", "--", "fastapi", "dev", "--host", "0.0.0.0", "main.py"]