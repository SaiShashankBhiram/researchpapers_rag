# Use full-featured Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Upgrade pip and install dependencies using Tsinghua mirror
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (adjust if app isn't named 'app' in main.py)
CMD ["uvicorn", "attention_yolo.main:app", "--host", "0.0.0.0", "--port", "8000"]