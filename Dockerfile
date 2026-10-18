FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use a reliable mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install CPU-only PyTorch FIRST (avoids 5+ GB of CUDA downloads)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt

COPY . .
CMD ["python", "-m", "runtime.benchmark"]