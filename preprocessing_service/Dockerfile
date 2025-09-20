FROM python:3.11-slim

WORKDIR /app

COPY requirements_preprocessing.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_preprocessing.txt

COPY . .

CMD ["uvicorn", "preprocessing_app:app", "--host", "0.0.0.0", "--port", "8003"]