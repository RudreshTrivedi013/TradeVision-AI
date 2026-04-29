FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed for sentiment analysis
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)" 2>/dev/null || true

# Copy project files
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY monitoring/ monitoring/
COPY models/ models/
COPY data/features/ data/features/

# Create directories that are gitignored but needed at runtime
RUN mkdir -p logs data/raw data/metadata

# Expose default ports (Render overrides via $PORT)
EXPOSE 8000 8501

# Default: run FastAPI with dynamic port from Render
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
