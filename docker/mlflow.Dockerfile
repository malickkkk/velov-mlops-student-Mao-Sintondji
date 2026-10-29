FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mlflow==3.16.1 \
    "psycopg[binary]>=3.2,<4"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]