PYTHON ?= python3
PKG = talk2me_speech

.PHONY: install test lint format train evaluate docker-up docker-down

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests scripts

format:
	black src tests scripts

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
