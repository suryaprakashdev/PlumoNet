FROM python:3.11-slim

WORKDIR /app

COPY . .

# 1. Install required system/OS libraries for OpenGL/OpenCV support
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install bentoml
RUN pip install -r requirements.txt

EXPOSE 8080

CMD bentoml serve service:LungNoduleService --host 0.0.0.0 --port ${PORT:-3000}

