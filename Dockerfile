FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.docker.txt .
ARG TORCH_VARIANT=cu124
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" && pip install --no-cache-dir -r requirements.docker.txt
COPY app ./app
COPY scripts ./scripts
COPY configs ./configs
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "2097152"]
