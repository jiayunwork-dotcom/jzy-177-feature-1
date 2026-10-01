FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DOA_DATA_DIR=/data

WORKDIR /app

# 依赖单独一层，便于利用构建缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 应用、实验与测试
COPY app ./app
COPY experiments ./experiments
COPY tests ./tests
COPY pytest.ini requirements.txt ./

# 会话数据目录（挂载点）
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 容器内由 uvicorn 监听；可由命令行参数覆盖
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
