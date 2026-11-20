PYTHON ?= python

.PHONY: color-test-images
# Regenerates the color test inputs and their references
color-test-images:
	$(PYTHON) scripts/makeColorTestImages.py
