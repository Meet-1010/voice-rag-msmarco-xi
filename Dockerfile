# Hugging Face Spaces, Docker SDK. Single container: API, static frontend, and an
# embedded Qdrant. No sidecar services, nothing to orchestrate.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/hf \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the encoder into the image. Downloading it on first request would put a
# multi-second stall in front of the first user and wreck the P100 story.
RUN python -c "import yaml; from index.embedder import Embedder; \
    Embedder(yaml.safe_load(open('config.yaml'))['embedder'])"

# Build the index at image build time so container start is just a process boot.
RUN python index/build_index.py --corpus data/corpus.deploy.jsonl \
    && python -c "import json; print(json.load(open('.artifacts/manifest.json')))"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
