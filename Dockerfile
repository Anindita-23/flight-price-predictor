# ── Stage 1: install dependencies into a virtualenv ──────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the virtualenv from builder (no build tools in final image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app.py predict.py ./
COPY templates/ templates/

# Model artefacts are mounted via docker-compose volume at runtime.
# Create the directory so Flask can load from it.
RUN mkdir -p model

EXPOSE 5000

# Run with production-friendly settings (no debug, bind to 0.0.0.0)
CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=5000)"]
