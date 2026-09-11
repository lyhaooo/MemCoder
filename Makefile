.PHONY: install test lint demo

install:
	python3 -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

demo:
	memcoder demo

