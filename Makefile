DOCKER_COMPOSE ?= docker compose

.PHONY: help up build down logs ps restart

help:
	@echo "Available targets:"
	@echo "  make up        # Build images and start the full stack"
	@echo "  make build     # Build images without starting containers"
	@echo "  make down      # Stop and remove containers"
	@echo "  make logs      # Tail logs from all services"
	@echo "  make ps        # List running containers"
	@echo "  make restart   # Restart the stack"

up:
	$(DOCKER_COMPOSE) up --build

build:
	$(DOCKER_COMPOSE) build

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f

ps:
	$(DOCKER_COMPOSE) ps

restart: down up
