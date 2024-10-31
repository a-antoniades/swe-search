.PHONY: build push deploy upgrade

build:
	@echo "Building Docker image..."
	@bash build_and_push.sh

push: build
	@echo "Image pushed as part of the build process."

deploy: push
	@echo "Deploying to Kubernetes..."
	@bash deploy.sh

upgrade: push
	@echo "Upgrading deployment..."
	@bash upgrade.sh

all: deploy