FROM python:3.11-slim
WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install D2 for diagram generation
RUN curl -fsSL https://d2lang.com/install.sh | sh -s --

# Python Dependencies 
# 1. Upgrade pip 
RUN pip install --no-cache-dir --upgrade pip
# 2. Install numpy
RUN pip install --no-cache-dir "numpy<2.0"
# 3. Install scipy with pinned version 
RUN pip install --no-cache-dir scipy==1.10.1
# 4. Install scikit-learn with pinned version 
RUN pip install --no-cache-dir scikit-learn==1.3.2
# 5. Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
# 6. Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
# Copy application code
COPY . .
EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
