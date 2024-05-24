.DEFAULT: help

SHELL := /bin/bash
VENV=venv
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip

help:
	@echo "sei lá"


venv:
	python3 -m venv $(VENV)
	. ./$(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt


run: venv
	$(PYTHON) src/__main__.py
