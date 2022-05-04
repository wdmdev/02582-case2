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

features:
	python src/features/build_features.py

visualize:
	python src/visualization/visualize.py

train:
	python src/models/train_model.py

predict:
	python src/models/predict_model.py --model $(model) --image_folder $(image_folder) --output $(output)

face_matching:
	python src/models/face_match_model.py --model $(model) --image_folder $(image_folder) --output $(output)

image_distances:
	python src/visualization/image_distances.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
