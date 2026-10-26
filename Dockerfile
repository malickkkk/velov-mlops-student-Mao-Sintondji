FROM python:3.12-slim AS builder

WORKDIR /app

ARG MODEL_VERSION=1.0.0
ARG GIT_COMMIT=dev

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN python -m velov.data --days 90 --stations 20 --out /tmp/velov_history.csv \
    && python -m velov.train --data /tmp/velov_history.csv --out /app/models --version "${MODEL_VERSION}" \
    && rm /tmp/velov_history.csv

FROM python:3.12-slim AS runtime

ARG MODEL_VERSION=1.0.0
ARG GIT_COMMIT=dev

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN useradd --create-home --user-group appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app/src ./src
COPY --from=builder --chown=appuser:appuser /app/models ./models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready').getcode() == 200 else 1)"

LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

CMD ["uvicorn", "velov.api.main:app", "--host", "0.0.0.0", "--port", "8000"]