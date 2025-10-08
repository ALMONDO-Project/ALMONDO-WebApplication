# ==============================
# Stage 1: Base Python image
# ==============================
FROM python:3.11-slim AS base

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    ALMONDO_MODEL_REPO="https://github.com/ALMONDO-Project/ALMONDO-Model.git" \
    CONFORMITY_REPO="https://github.com/GiulioRossetti/conformity.git" \
    CONFORMITY_FALLBACK_REPO="https://github.com/verdiana01/conformity.git"

# Create working directory
WORKDIR /home/almondo

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for efficient caching
COPY requirements.txt ./requirements.txt

# Install dependencies safely
RUN pip install --upgrade pip setuptools wheel \
    # Remove editable installs (causing Docker path issues)
    && sed -i '/^-e /d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    # Install the ALMONDO model dependency
    && pip install --no-cache-dir "git+${ALMONDO_MODEL_REPO}" \
    # Try installing the original conformity repo first
    && (pip install --no-cache-dir "git+${CONFORMITY_REPO}" \
        || pip install --no-cache-dir "git+${CONFORMITY_FALLBACK_REPO}") \
    # Add gunicorn for production serving
    && pip install --no-cache-dir gunicorn \
    && rm -rf /root/.cache/pip /tmp/pip*

# ==============================
# Stage 2: Application setup
# ==============================
FROM base AS runtime

WORKDIR /home/almondo

# Copy application source
COPY . .

# Expose the app port
EXPOSE 8000

# Default command for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
