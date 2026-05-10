# ── Base image ────────────────────────────────────────────────────────────────
# CUDA 12.1 devel + cuDNN 8, Ubuntu 22.04.
# 'devel' is required (not 'runtime') because CTranslate2 dynamically loads
# libcublas.so.12 and libcublasLt.so.12 at inference time — these are only
# present in the devel image.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Store HuggingFace / CTranslate2 model weights at a known path.
    HF_HOME=/models \
    CT2_VERBOSE=1

# ── System dependencies ───────────────────────────────────────────────────────
# libsndfile1   → required by soundfile (WAV reading)
# ffmpeg        → fallback audio decoding (mp3 / m4a etc.)
# python3.11    → matches faster-whisper's recommended Python version
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Pre-download model weights ────────────────────────────────────────────────
# Downloads happen at BUILD time so the container starts instantly.
# Models are stored at /models (mapped to HF_HOME).
# We load on CPU here — no GPU is available during docker build.
RUN python3 -c "\
from faster_whisper import WhisperModel; \
print('>>> Downloading large-v3 ...'); \
WhisperModel('large-v3', device='cpu', download_root='/models'); \
print('>>> Downloading large-v2 ...'); \
WhisperModel('large-v2', device='cpu', download_root='/models'); \
print('>>> All models downloaded.')"

# ── Application source ────────────────────────────────────────────────────────
COPY src/ .

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 5001

# Single worker — both GPU models share one CUDA context. Multiple workers
# would require a model-per-worker strategy and exceed T4 VRAM.
CMD ["uvicorn", "asr_server:app", \
     "--host", "0.0.0.0", \
     "--port", "5001", \
     "--workers", "1", \
     "--log-level", "info"]