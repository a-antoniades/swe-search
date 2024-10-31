#!/bin/bash

# Set variables
IMAGE_NAME="aorwall/moatless-vizualize-tree"
TAG=$(git rev-parse --short HEAD)

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .
docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest

# Push the Docker image
echo "Pushing Docker image..."
docker push ${IMAGE_NAME}:${TAG}
docker push ${IMAGE_NAME}:latest

echo "Build and push completed successfully."
echo "Image: ${IMAGE_NAME}:${TAG}"