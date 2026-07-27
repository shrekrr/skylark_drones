# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install GDAL system deps needed by rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cacheable layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source only — rasters and model are mounted at runtime
COPY solution/ ./solution/
COPY schemas/  ./schemas/

# No data/ or model/ directories — they are mounted at /input and /model at runtime
ENTRYPOINT ["python", "-m", "solution.infer"]
