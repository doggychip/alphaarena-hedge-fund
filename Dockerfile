FROM python:3.11-slim
WORKDIR /app
# Install build deps
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi
COPY . .
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
