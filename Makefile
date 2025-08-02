# Makefile for Tweet Classification MLOps Project


.PHONY: help build run lint test clean orchestrate retrain clean-models

help:
	@echo "Available targets:"
	@echo "  build        Build the Docker image."
	@echo "  run          Run the FastAPI server in Docker."
	@echo "  lint         Lint Python code with flake8."
	@echo "  test         Run all unit tests."
	@echo "  retrain	  Retrain the model with svn tfidf."
	@echo "  clean-models Remove all MLflow models."
	@echo "  orchestrate  Run the Prefect workflow pipeline."
	@echo "  clean        Remove Python cache and Docker images."
orchestrate:
	python src/prefect_flow.py

clean-models:
	find . -name "*.pkl" ! -name "*20250801*" -delete

build:
	docker build -t tweet-classification-api .

run:
	docker compose up -d

lint:
	flake8 src/ *.py

test:
	pytest tests/ || python3 -m unittest discover -s tests

retrain:
	python src/train_svm.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	docker rmi tweet-classification-api || true
