FROM python:3.12-slim

# Prevent Python from creating .pyc files
# and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install --no-cache-dir uv

# Install project dependencies
RUN uv sync --frozen

# Copy application files
COPY app.py ./

COPY src ./src

COPY models ./models

COPY configs ./configs

# Create runtime directories
RUN mkdir -p snapshots logs

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]