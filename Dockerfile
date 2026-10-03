FROM python:3.12-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set model cache directories to /tmp to avoid runtime permission issues
ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

# Set a thread budget to avoid thread congestion in container environments
ENV OMP_NUM_THREADS=4

# Keep the pip cache in the deps_cache volume, not the image layer, so it survives
# container replacement and never bloats the image
ENV PIP_CACHE_DIR=/opt/deps/pip-cache

# Install system dependencies required for document processing (libgl for OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Word dir

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download default ML models so the container works fast and offline
RUN docling-tools models download

COPY . .

# ENTRYPOINT, not CMD: `docker compose run app ...` overrides CMD but not
# ENTRYPOINT, so the dependency sync below runs for every invocation. Source is
# bind-mounted over /app by compose, so editing .py files needs no rebuild and
# editing requirements.txt needs no rebuild either.
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]

