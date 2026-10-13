FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system dependencies (libgomp1 is required by LightGBM and XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifact
COPY model_architecture.py .
COPY model.pkl .
COPY main.py .

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Run FastAPI via Uvicorn, dynamically binding to the PORT environment variable
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
