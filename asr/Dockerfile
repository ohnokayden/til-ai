FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='float32', download_root='/app/models'); WhisperModel('large-v2', device='cpu', compute_type='float32', download_root='/app/models')"

COPY src .

HEALTHCHECK --interval=20s --timeout=15s --start-period=300s --retries=15 CMD curl -f http://localhost:5001/health || exit 1

CMD ["uvicorn", "asr_server:app", "--port", "5001", "--host", "0.0.0.0"]
