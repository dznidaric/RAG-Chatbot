#image for build
FROM python:3.10 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# image for app from build
FROM python:3.10-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local

COPY . .
ENV PATH="/root/.local/bin:$PATH"
CMD ["python", "main.py"]
