# Makefile for Tweet Classification MLOps Project


.PHONY: help build run lint test clean orchestrate

help:
	@echo "Available targets:"
	@echo "  build        Build the Docker image."
	@echo "  run          Run the FastAPI server in Docker."
	@echo "  lint         Lint Python code with flake8."
	@echo "  test         Run all unit tests."
	@echo "  orchestrate  Run the Prefect workflow pipeline."
	@echo "  clean        Remove Python cache and Docker images."
orchestrate:
	python src/prefect_flow.py

build:
	docker build -t tweet-classification-api .

run:
	docker run --rm -p 8000:8000 tweet-classification-api

lint:
	flake8 src/ *.py

test:
	pytest tests/ || python3 -m unittest discover -s tests

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	docker rmi tweet-classification-api || true
