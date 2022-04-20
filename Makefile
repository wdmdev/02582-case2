lint:
	flake8 src

format:
	black src
	isort src

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

dataset:
	python src/data/make_data.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
