PYTHON_INTERPRETER = python3

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -U -r requirements.txt

convert-template-ipynb-to-md:
	jupytext --to md templates/*.ipynb

convert-template-md-to-ipynb:
	jupytext --to ipynb templates/*.md

create-ques-sol-mds:
	$(PYTHON_INTERPRETER) create_mds.py

convert-ques-sol-md-to-ipynb:
	jupytext --to ipynb --execute  questions/*.md
	jupytext --to ipynb --execute  solutions/*.md

remove-no-cell-exec-artifact:
	gawk -i inplace '!/%%script echo /' questions/*.ipynb

convert-ques-sol-ipynb-to-pdf:
	jupyter nbconvert --to pdf questions/*.ipynb
	jupyter nbconvert --to pdf solutions/*.ipynb

convert-full: create-ques-sol-mds \
 	convert-ques-sol-md-to-ipynb \
	remove-no-cell-exec-artifact \
	convert-ques-sol-ipynb-to-pdf
