.PHONY: install all generate pipeline test lint clean

install:
	pip install -r requirements.txt

all: pipeline

generate:
	python -m src.generate_data

pipeline:
	python -m src.run_pipeline

fast:
	python -m src.run_pipeline --skip-generate --no-figures

test:
	pytest -q

clean:
	rm -rf data/raw/*.csv data/processed/* reports/*.csv reports/*.json reports/figures/*.png powerbi/data/*.csv
