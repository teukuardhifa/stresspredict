FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# gunakan gunicorn dan bind ke $PORT
CMD ["sh", "-c", "gunicorn app:app -b 0.0.0.0:${PORT} --workers 2 --threads 2"]
