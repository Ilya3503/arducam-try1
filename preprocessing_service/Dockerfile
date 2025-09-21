FROM python:3.11-slim

WORKDIR /app

COPY requirements_preprocessing.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_preprocessing.txt

COPY . .

CMD ["uvicorn", "preprocessing_app:app", "--host", "0.0.0.0", "--port", "8003"]
