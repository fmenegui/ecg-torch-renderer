.PHONY: test docs check examples

test:
	python -m unittest discover -s tests -v

docs:
	python documentation/build.py
	python documentation/validate.py

examples:
	python examples/sequential.py
	python examples/segmentation.py --steps 3

check: test docs examples
