# 1. Start with Python installed
FROM python:3.13-slim

# 2. Set the working directory inside the image
WORKDIR /app

# 3. Install Linux libraries needed by FAISS
RUN apt-get update && apt-get install -y \
    --no-install-recommends libgomp1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy backend code
COPY backend/ ./backend/

# 6. Document the API port
EXPOSE 8000

# 7. Start FastAPI when the container runs
CMD ["uvicorn", "backend.app.main:app", \
    "--host", "0.0.0.0", "--port", "8000"]