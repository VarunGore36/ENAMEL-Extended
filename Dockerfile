FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies (minimal)
RUN pip install --no-cache-dir numpy matplotlib

# Set environment for reproducible timing
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
ENTRYPOINT ["python3", "scripts/evaluate.py"]
CMD ["run", "--problems", "enamel_ext/data/cache/problems.json", "--solutions", "enamel_ext/data/cache/evalplus_solutions.json", "--keep-going", "--quiet"]
