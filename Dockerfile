# 使用官方 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# Cloud Run 会传入 PORT 环境变量，Flask 必须监听它
ENV PORT=8080

# 运行你的 main.py，而不是用 flask CLI（更稳）
CMD ["python", "main.py"]
