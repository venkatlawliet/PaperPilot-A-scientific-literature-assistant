FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit config to prevent error
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
