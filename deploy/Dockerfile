FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml requirements-api.lock ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir -r requirements-api.lock && python -m pip install --no-cache-dir '.[inference]'
ENV MOE_PROJECT_ROOT=/app MOE_MODEL_ROOT=/models MOE_HOST=127.0.0.1 MOE_PORT=8510 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
EXPOSE 8510
HEALTHCHECK --start-period=180s --interval=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8510/health',timeout=5)"
CMD ["python", "-m", "moe_router"]
