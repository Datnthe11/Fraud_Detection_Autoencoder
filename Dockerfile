FROM python:3.12-slim

LABEL author="DAT"

WORKDIR /app

COPY requirements.txt .

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn (thư mục app)
COPY app/ ./app

# Mở cổng 8888
EXPOSE 8888

# Chạy FastAPI app (module: app.main, biến: app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
