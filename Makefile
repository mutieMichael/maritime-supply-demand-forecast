PYTHON ?= python3

.PHONY: install data train test app notebook all

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PYTHON) scripts/generate_data.py

train: data
	$(PYTHON) scripts/train_evaluate.py

test:
	$(PYTHON) -m pytest

app:
	$(PYTHON) -m streamlit run app.py --server.headless true

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/demand_forecast_walkthrough.ipynb --inplace

all: install data train test
