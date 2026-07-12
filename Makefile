.PHONY: install train run test cov lint docker demo

install:
	pip install -r requirements.txt

train:
	python -m awaf.train

demo:
	python -m awaf.demo_feedback

run:
	uvicorn awaf.api:app --reload --port 8000

test:
	pytest

cov:
	pytest --cov=awaf --cov-report=term-missing

lint:
	ruff check awaf tests

docker:
	docker compose up --build
