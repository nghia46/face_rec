FROM python:3.11-slim as builder
RUN apt-get update && apt-get install -y gcc g++ python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
WORKDIR /app

EXPOSE 8000

COPY . .
ENV PATH="/root/.local/bin:$PATH"
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]