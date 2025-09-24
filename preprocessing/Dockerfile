FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libegl1 \
    libx11-6 \
    libosmesa6-dev \
    libglib2.0-0 \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements_preprocessing.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_preprocessing.txt

# Копируем код
COPY . .

CMD ["uvicorn", "preprocessing_app:app", "--host", "0.0.0.0", "--port", "8003"]
