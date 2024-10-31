#!/bin/bash

# Set variables
IMAGE_NAME="aorwall/moatless-vizualize-tree"
TAG=$(git rev-parse --short HEAD)

# Update the deployment YAML with the new image tag
sed -i.bak "s|image: ${IMAGE_NAME}:.*|image: ${IMAGE_NAME}:${TAG}|" k8s-deployment.yaml

# Apply the updated deployment
echo "Upgrading deployment..."
kubectl apply -f k8s-deployment.yaml

# Wait for the rollout to complete
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/streamlit-moatless

echo "Upgrade completed successfully."
echo "New image: ${IMAGE_NAME}:${TAG}"