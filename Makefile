PY ?= .venv/bin/python

.PHONY: setup data sessions fleet figures test lint all

setup:            ## create the virtual environment and install the package
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

data:             ## download public raw data (~160 MB) and build processed tables
	$(PY) -m evfleet.data.download
	$(PY) -m evfleet.data.palo_alto
	$(PY) -m evfleet.data.tlc

sessions:         ## Palo Alto session models, segmentation, fee analysis, utilization
	$(PY) scripts/run_session_analysis.py

fleet:            ## robotaxi charging experiments (about 10 minutes on 6 cores)
	$(PY) scripts/run_fleet_experiments.py

figures:          ## regenerate every figure in reports/figures from reports/results
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

all: data sessions fleet figures
