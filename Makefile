PYTHON ?= python3

.PHONY: test test-unit test-e2e lint

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-unit:
	$(PYTHON) -m unittest tests.test_download_core tests.test_download_progress -v

test-e2e:
	$(PYTHON) -m unittest tests.test_download_e2e -v

lint:
	$(PYTHON) -m py_compile aim.py

