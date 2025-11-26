FROM python:3.11-slim

# 安装基础依赖（faiss、tokenizers、torch 这些都需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 安装 Python包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# Cloud Run 默认使用 $PORT
ENV PORT=8080

# 使用 gunicorn 而不是 flask run（🔥 必须）
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app
