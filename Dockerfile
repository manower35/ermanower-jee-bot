# ═══════════════════════════════════════════════════════════════════════════
# ErManower JEE Architecture Planning — Multi-Stage Production Dockerfile
# ═══════════════════════════════════════════════════════════════════════════
# Lightweight image based on python:3.11-slim.
# Unbuffered stdout/stderr for real-time cloud logging.
# ═══════════════════════════════════════════════════════════════════════════

# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some C-extension wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

# Disable Python output buffering for real-time cloud logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY main.py .
COPY utils.py .
COPY database.py .
COPY crew_orchestrator.py .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Health check — ensure Python can import the app
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import main; print('ok')" || exit 1

# Entry point
CMD ["python", "main.py"]
