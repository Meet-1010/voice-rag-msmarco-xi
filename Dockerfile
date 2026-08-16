# Single container: API, static frontend, and an embedded Qdrant. No sidecars.
# Runs on Cloud Run, Render, Railway or HF Spaces unchanged.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create the workdir as root and hand it to uid 1000 before switching user.
# WORKDIR creates missing directories as root even when USER is already set, so
# without this the app cannot create .qdrant/ inside its own working directory.
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/hf
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .

# Bake the encoder in. Downloading it on first request would put a multi-second
# stall in front of the first user and wreck the P100 story.
RUN python -c "import yaml; from index.embedder import Embedder; \
    Embedder(yaml.safe_load(open('config.yaml'))['embedder'])"

# Prefer the index committed alongside the image. Rebuilding it here costs ~19
# minutes on a cloud builder (16 chunks/s vs 205 on a dev machine), and shipping
# the prebuilt one also guarantees the deployed index is byte-identical to the
# one the benchmarks and guardrail calibration were measured against.
# Falls back to building from the corpus so the image is still reproducible from
# source alone.
RUN if [ ! -f .artifacts/manifest.json ]; then \
      echo "no prebuilt index, building from corpus"; \
      EMBED_THREADS=8 python index/build_index.py --corpus data/corpus.deploy.jsonl; \
    else echo "using prebuilt index"; fi

# Fail at build time, not at 3am on a cold start: confirms the Qdrant storage
# actually loads on this platform and that BM25 and the chunk map came along.
RUN python -c "\
import json, yaml, sys; \
from index.store import VectorStore; \
from pathlib import Path; \
cfg = yaml.safe_load(open('config.yaml')); \
s = VectorStore(cfg, Path('.')); \
n = s.count(); \
m = json.load(open('.artifacts/manifest.json')); \
assert Path('.artifacts/bm25.pkl').exists(), 'bm25 index missing'; \
assert n == m['chunks'], f'index has {n} points, manifest says {m[\"chunks\"]}'; \
print(f'index verified: {n} points across {s.languages()}, strategy={m[\"strategy\"]}')"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:${PORT:-7860}/health || exit 1

# Shell form so $PORT expands: Cloud Run, Render and Railway all inject their own
# port and ignore EXPOSE. Falls back to 7860 for local.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}
