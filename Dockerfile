# Optimized PyTorch Docker Image for AI Text Detection
# This Dockerfile includes multiple optimizations for better PyTorch performance

ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM=linux/amd64

# Use a slim base image with Python
FROM --platform=${TARGETPLATFORM} python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies with optimizations
# - Install build essentials for optimized wheel builds
# - Install Intel MKL for CPU optimization (if on Intel hardware)
# - Install OpenBLAS as alternative high-performance BLAS library
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libgomp1 \
    default-jre \
    # BLAS/LAPACK libraries for optimized linear algebra
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    # Additional optimization libraries
    libomp-dev \
    # Cleanup to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with CPU optimizations
# Using CPU-optimized PyTorch for smaller image size and faster CPU inference
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install PyTorch with CPU optimizations (includes MKL)
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url ${TORCH_INDEX_URL} && \
    # Install remaining dependencies
    pip install --no-cache-dir -r requirements.txt

# Download required spaCy models
RUN python -m spacy download xx_sent_ud_sm && \
    python -m spacy download en_core_web_sm

# Pre-download LanguageTool server files for English
# This reduces startup time by caching language packs
RUN printf 'import language_tool_python\nlanguages = [\n    "en-US", "en-GB", "en-CA", "en-AU", "en-NZ", "en-ZA"\n]\nfor lang in languages:\n    try:\n        language_tool_python.LanguageTool(lang)\n        print(f"Downloaded language pack for {lang}")\n    except Exception as e:\n        print(f"Skipping {lang}: {e}")\n' > /tmp/download_languages.py && \
    python /tmp/download_languages.py && \
    rm /tmp/download_languages.py

# Copy application code
COPY app.py .
COPY api.py .
COPY copyscape_helper.py .

# Create model and uploads directories
RUN mkdir -p /app/model /app/uploads

# Expose ports
EXPOSE 7860 8000

# Set environment variables for maximum PyTorch performance
ENV PYTHONUNBUFFERED=1 \
    # Transformers optimizations
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    # Disable tokenizers parallelism (important for multiprocessing)
    TOKENIZERS_PARALLELISM=false \
    # PyTorch CPU optimizations
    # Use all available CPU cores for intra-op parallelism (matrix operations)
    OMP_NUM_THREADS="" \
    MKL_NUM_THREADS="" \
    # Inter-op parallelism threads (operations between layers)
    # Set this based on your CPU cores (leave empty to auto-detect)
    TORCH_NUM_THREADS="" \
    # OpenMP optimizations
    OMP_WAIT_POLICY=PASSIVE \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores \
    # Intel MKL optimizations (if using Intel CPU)
    MKL_THREADING_LAYER=GNU \
    MKL_DYNAMIC=TRUE \
    # Memory allocator optimizations
    MALLOC_TRIM_THRESHOLD_=100000 \
    MALLOC_MMAP_THRESHOLD_=100000 \
    # Python optimizations
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    # Reduce HuggingFace verbosity
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Run the application
CMD ["python", "-u", "app.py"]

