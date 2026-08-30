# ReRouteHer It1 backend image.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # never touch the Hugging Face Hub at runtime; the model is vendored below
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_MODEL=models/all-MiniLM-L6-v2

WORKDIR /app

# Python deps first for layer caching.
# Install CPU-only torch up front so sentence-transformers does not pull the ~1GB
# NVIDIA CUDA wheels (useless on CPU); the rest then sees torch already satisfied.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# spaCy English model (fetched from spaCy's own release, not Hugging Face).
RUN python -m spacy download en_core_web_sm

# Vendored embedding model (checked into the repo, ~87MB). Copied straight in, so the
# build performs no Hugging Face download and startup loads it from this path.
COPY models ./models

# Vendored TF-IDF occupation classifier (~25MB). Baked in so Tier 1 works on hosts
# that do not mount the compose volumes (e.g. Coolify).
COPY ml ./ml

# App code (db/, tests/ excluded via .dockerignore where not needed at runtime).
COPY app ./app

# Run as a non-root user (matches workspace convention).
RUN useradd -u 3000 -m appuser && chown -R 3000:3000 /app
USER 3000

# Network interface (8080; 8000 is taken by Coolify on the host).
EXPOSE 8080

# --no-access-log: RequestLoggingMiddleware logs safe metadata, never CV bodies.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
