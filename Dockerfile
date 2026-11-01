FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: editing application code must not invalidate the
# multi-hundred-MB torch/docling layer.
COPY requirements.txt requirements-documents.txt ./
RUN pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-documents.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . \
    && useradd --create-home --uid 10001 repopilot \
    && mkdir -p /app/.repopilot /workspace \
    && chown -R repopilot:repopilot /app/.repopilot /workspace

USER repopilot

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "repopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
