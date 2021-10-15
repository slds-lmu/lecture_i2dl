PYTHON_INTERPRETER = python3

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -U -r requirements.txt

convert-ipynb-to-md:
	jupytext --to md $(FILE)

convert-md-to-ipynb:
	jupytext --to ipynb $(FILE)

