FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY middleware ./middleware
COPY run.py config.example.toml ./
RUN uv sync --frozen --no-dev \
    && cp config.example.toml config.toml \
    && sed -i 's/host = "127.0.0.1"/host = "0.0.0.0"/' config.toml

EXPOSE 8787

CMD ["python", "run.py"]
