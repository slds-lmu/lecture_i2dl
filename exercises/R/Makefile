.PHONY: all

# must create pdf manually first
# touch generated/lab{1..11}-{questions,solutions}.pdf -t 0001010000
all: make-handouts.py $(wildcard generated/lab*-solutions.pdf) $(wildcard generated/lab*-questions.pdf) generated/lab0.pdf

generated/%.png: %.png make-handouts.py
	cp %.png generated/

generated/lab%-questions.Rmd: lab%-template.Rmd make-handouts.py
	python make-handouts.py lab$*-template.Rmd

generated/lab%-questions.pdf: generated/lab%-questions.Rmd make-handouts.py
	Rscript -e 'rmarkdown::render("generated/lab$*-questions.Rmd")'

generated/lab%-solutions.Rmd: lab%-template.Rmd make-handouts.py
	python make-handouts.py lab$*-template.Rmd

generated/lab%-solutions.pdf: generated/lab%-solutions.Rmd make-handouts.py
	Rscript -e 'rmarkdown::render("generated/lab$*-solutions.Rmd")'

generated/lab0.pdf: lab0.Rmd make-handouts.py
	Rscript -e 'rmarkdown::render("lab0.Rmd")'
	mv lab0.pdf generated/
