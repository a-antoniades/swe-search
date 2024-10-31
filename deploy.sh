#!/bin/bash

# Set variables
IMAGE_NAME="aorwall/moatless-vizualize-tree"
TAG=$(git rev-parse --short HEAD)

# Update the deployment YAML with the new image tag
sed -i.bak "s|image: ${IMAGE_NAME}:.*|image: ${IMAGE_NAME}:${TAG}|" k8s-deployment.yaml

# Apply Kubernetes configurations
echo "Deploying to Kubernetes..."
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml

echo "Deployment completed successfully."