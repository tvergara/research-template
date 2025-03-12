.PHONY: train test lint

train:
	poetry run python -m src.train

test:
	poetry run pytest

lint:
	poetry run black --check .
	poetry run isort --check-only .
	poetry run flake8 .

check: lint test

