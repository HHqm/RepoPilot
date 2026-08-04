FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir "torch>=2.2,<3" \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[documents]"

EXPOSE 8000
CMD ["uvicorn", "repopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
