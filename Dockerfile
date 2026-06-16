FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir bentoml \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD bentoml serve service:LungNoduleService --host 0.0.0.0 --port ${PORT:-3000}
