# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
     build-essential  graphviz-dev  graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

COPY poetry.lock pyproject.toml /app/

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the current directory contents into the container at /app
COPY moatless /app/moatless
COPY examples_app.py /app
COPY examples /app/examples

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
CMD ["poetry", "run", "streamlit", "run", "examples_app.py", "--server.port=8501", "--server.address=0.0.0.0"]