FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Copy scripts, worker & configs
COPY scripts/ /app/scripts/
COPY worker/ /app/worker/
COPY config.yaml /app/config.yaml
COPY .env.example /app/.env.example

# Python deps
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart pyyaml \
    requests httpx \
    'markitdown[pdf]'

# Scrapling + Patchright + Chromium browser (Tier 2 stealth fetcher)
RUN pip install --no-cache-dir "scrapling[fetchers]>=0.4.7" patchright msgspec && \
    python -m playwright install-deps chromium && \
    python -m playwright install chromium && \
    python -m patchright install-deps chromium && \
    python -m patchright install chromium && \
    rm -rf /root/.cache/ms-playwright/.temp /tmp/*

# Data volume (shared with dashboard)
VOLUME ["/app/data"]

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3001/api/health || exit 1

CMD ["python", "-m", "worker.server"]
