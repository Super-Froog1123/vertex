# 使用官方镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 只在 requirements.txt 变动时安装依赖（加快构建）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 然后再拷贝主代码
COPY . .

# 其他配置
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
EXPOSE 8080
CMD ["flask", "run"]
