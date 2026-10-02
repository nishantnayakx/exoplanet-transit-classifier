import os

# Gunicorn configuration for production deployment (Render / container)
port = os.environ.get("PORT", "8050")
bind = f"0.0.0.0:{port}"

# Single worker with multi-threading to stay well within 512MB RAM
workers = 1
threads = 4
worker_class = "gthread"

# Set timeout to prevent premature worker timeouts
timeout = 120
keepalive = 5

# CRITICAL: Do NOT preload application before forking to prevent PyTorch OpenMP deadlock
preload_app = False
