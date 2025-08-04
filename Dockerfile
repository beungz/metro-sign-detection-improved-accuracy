FROM python:3.10-slim

# Avoid interactive prompts and reduce image size
ENV DEBIAN_FRONTEND=noninteractive

# Create working directory
WORKDIR /app

# Install system-level dependencies for OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip==24.0 setuptools wheel
RUN pip install -r requirements.txt

# Selectively copy only inference code
COPY main.py ./main.py
COPY scripts/ ./scripts/
COPY models/ ./models/
# Create an empty folder "data"
RUN mkdir -p data

# Expose port
EXPOSE 8080

#Run app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]