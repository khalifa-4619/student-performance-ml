# =====================================================================
# STAGE 1: THE BUILDER
# =====================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Disable Python writing bytecode (.pyc) to disk and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for compiling wheels if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies directly into a local folder space
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =====================================================================
# STAGE 2: THE FINAL RUNTIME ELASTIC CORE
# =====================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force Python to look at our compiled site-packages from the builder stage
ENV PATH=/home/mluser/.local/bin:$PATH

# 1. SECURITY LOCKDOWN: Create a limited system user and group
# Production containers should NEVER run as root to mitigate container breakout exploits.
RUN groupadd -g 10001 mlgroup && \
    useradd -u 10001 -g mlgroup -m -s /bin/bash mluser

# 2. Extract exclusively the compiled dependencies from the builder stage
COPY --from=builder /root/.local /home/mluser/.local

# 3. Copy application files and adjust ownership to our non-root user
COPY --chown=mluser:mlgroup app/ /app/backend/app/

# 4. Swap execution privileges over to the secure restricted account
USER mluser

# Expose the network port inside the container cluster mesh
EXPOSE 8000

# Execute the application using Uvicorn bound to all interfaces
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]