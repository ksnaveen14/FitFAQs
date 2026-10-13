# ── FitRAG Dockerfile ────────────────────────────────────────────────────────
# Works for: Railway, Render, Fly.io, or any container host.
#
# Build:  docker build -t fitrag .
# Run:    docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... fitrag
# Open:   http://localhost:8501

FROM python:3.11-slim

# System deps for faiss-cpu and pypdf
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache — only reruns if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Streamlit config — disable telemetry, set port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# Health check (Railway/Render use this to know the app is up)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
