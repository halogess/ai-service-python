# =========================================
# Stage 1: Builder (install dependencies)
# =========================================
FROM python:3.11-slim AS builder

# Install build tools + curl for uv installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using official installer (bypass ghcr.io)
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

ARG UV_HTTP_TIMEOUT=10000
ENV UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT} \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

# Cache mount for blazing fast repeated builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# =========================================
# Stage 2: Runtime (minimal)
# =========================================
FROM python:3.11-slim AS runtime

# Only runtime libs needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only installed packages (no build tools)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root user (security best practice)
RUN useradd --create-home --shell /bin/false --uid 1001 appuser
USER appuser

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser .env .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-u", "src/main.py"]