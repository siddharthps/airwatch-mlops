# Use slim Python 3.10 base image
FROM python:3.10-slim

# Set workdir inside container
WORKDIR /app

# Install system dependencies and clean up in single layer
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv
RUN pip install --upgrade pip && pip install uv

# Copy all project files (needed for -e . in requirements.txt)
COPY . .

# Install Python dependencies using uv
RUN uv pip install --system -r requirements.txt

# Set Python path to ensure modules can be found
ENV PYTHONPATH=/app

# Run both flows sequentially as default (can be overridden)
CMD ["sh", "-c", "python flows/inference_data_preparation.py && python flows/model_inference.py"]
