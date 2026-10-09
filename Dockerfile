FROM python:3.12-slim
 
WORKDIR /app
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
 
COPY requirements.txt .
 
RUN pip install --no-cache-dir -r requirements.txt
 
COPY src ./src

COPY models ./models
 
RUN useradd --create-home appuser \
&& chown -R appuser:appuser /app
 
USER appuser
 
EXPOSE 8000
 
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
 
CMD ["uvicorn", "velov.api.main:app", "--host", "127.0.0.1", "--port", "8000"]
 