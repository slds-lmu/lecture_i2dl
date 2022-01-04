.PHONY: all

# must create pdf manually first
# touch generated/R-lab-{1..11}-{questions,solutions}.pdf -t 0001010000
all: make-handouts.py $(wildcard generated/R-lab-*-solutions.pdf) $(wildcard generated/R-lab-*-questions.pdf) generated/lab0.pdf

generated/%.png: %.png make-handouts.py
	cp %.png generated/

generated/R-lab-%-questions.Rmd: R-lab-%-template.Rmd make-handouts.py
	python make-handouts.py R-lab-$*-template.Rmd

generated/R-lab-%-questions.pdf: generated/R-lab-%-questions.Rmd make-handouts.py
	Rscript -e 'rmarkdown::render("generated/R-lab-$*-questions.Rmd")'

generated/R-lab-%-solutions.Rmd: R-lab-%-template.Rmd make-handouts.py
	python make-handouts.py R-lab-$*-template.Rmd

generated/R-lab-%-solutions.pdf: generated/R-lab-%-solutions.Rmd make-handouts.py
	Rscript -e 'rmarkdown::render("generated/R-lab-$*-solutions.Rmd")'

generated/lab0.pdf: lab0.Rmd make-handouts.py
	Rscript -e 'rmarkdown::render("lab0.Rmd")'
	mv lab0.pdf generated/
