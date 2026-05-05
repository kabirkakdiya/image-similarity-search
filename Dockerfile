FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . /app

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
