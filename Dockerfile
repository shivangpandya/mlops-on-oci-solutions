ARG BASE_IMAGE=docker.io/library/python:3.12-slim@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface REVIEWOPS_MODEL_DIR=/app/checkpoint \
    OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements-serving.txt ./
RUN python -m pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-serving.txt \
    && python -m pip check \
    && python -c "import torch; assert torch.version.cuda is None"
COPY app.py review_app.py review_inference.py review_remote.py release.json build-manifest.json ./
COPY static/ ./static/
COPY checkpoint/ ./checkpoint/
LABEL org.opencontainers.image.title="ReviewOps" \
    io.reviewops.model.version="2" \
    io.reviewops.mlflow.run="a42731d0210a442a8b6d687f17d998c9" \
    io.reviewops.model.revision="714eb0fa89d2f80546fda750413ed43d93601a13"
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
