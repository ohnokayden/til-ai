FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src .
RUN python download_models.py

HEALTHCHECK --interval=20s --timeout=15s --start-period=600s --retries=20 CMD curl -f http://localhost:5001/health || exit 1

CMD ["uvicorn", "asr_server:app", "--port", "5001", "--host", "0.0.0.0"]
