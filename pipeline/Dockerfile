FROM python:3.11-slim

WORKDIR /app

COPY requirements_pipeline.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_pipeline.txt

COPY . .

CMD ["uvicorn", "pipeline_app:app", "--host", "0.0.0.0", "--port", "8001"]
