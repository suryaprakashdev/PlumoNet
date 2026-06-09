FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install bentoml
RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["sh","-c","bentoml serve service:LungNoduleService --host 0.0.0.0 --port ${PORT:-8080}"]
