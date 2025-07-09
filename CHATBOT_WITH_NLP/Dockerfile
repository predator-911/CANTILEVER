# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create streamlit config directory
RUN mkdir -p ~/.streamlit

# Create streamlit config file
RUN echo "\
[general]\n\
email = \"\"\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
address = 0.0.0.0\n\
\n\
[theme]\n\
primaryColor = \"#FF6B6B\"\n\
backgroundColor = \"#FFFFFF\"\n\
secondaryBackgroundColor = \"#F0F2F6\"\n\
textColor = \"#262730\"\n\
" > ~/.streamlit/config.toml

# Create secrets file for caching
RUN echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/secrets.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
