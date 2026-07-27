# syntax=docker/dockerfile:1
FROM python:3.11-slim

# libgomp1: required by onnxruntime for OpenMP multi-threading
# libexpat1: required by some rasterio/GDAL wheel internals
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cacheable layer)
# rasterio wheels bundle GDAL -- no system GDAL installation needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source only -- rasters and model are mounted at runtime
COPY solution/ ./solution/
COPY schemas/  ./schemas/

# Evaluator appends: --manifest /input/manifest.json --model /model/gcp_pose.onnx --output /output/predictions.json
ENTRYPOINT ["python", "-m", "solution.infer"]
