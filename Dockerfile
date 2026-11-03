# CPU image for the Streamlit UI and CLI pipeline.
# Models download into AUTISM_CACHE_DIR on first run (mount a volume).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTISM_CACHE_DIR=/cache \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    SENTENCE_TRANSFORMERS_HOME=/cache/hf/sentence-transformers \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheel first so `torch` in requirements.txt does not pull a CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch

COPY requirements.txt .
RUN grep -vE '^[[:space:]]*torch([=<>]|$)' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir -r /tmp/requirements.notorch.txt \
    && rm /tmp/requirements.notorch.txt

COPY . .
RUN mkdir -p /cache /app/data /app/outputs \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /cache

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "ui.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
