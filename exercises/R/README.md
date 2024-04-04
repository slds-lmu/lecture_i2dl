# Introduction to Deep Learning - R Labs

This directory contains R based versions of the lab sessions.

## DEVELOPER: How to get started

The introduction to the lab is rendered with the following command:

```shell
make generated/lab0.html
```

You can directly edit the template`.Rmd` files. To turn them into pdfs use

- `make generated/R-lab-{number}-questions.pdf` for the questions
- `make generated/R-lab-{number}-solutions.pdf` for the solutions

To get the `.Rmd` files use

- `make generated/R-lab-{number}-questions.Rmd` for the questions
- `make generated/R-lab-{number}-solutions.Rmd` for the solutions

Some tags in the template lead to different behavior in question and solution
versions:

- Within the R code: `#!hwbegin <optional text>` and `#!hwend`: Everything in between does two tags is treated as
homework for the student. The content in this block is only included in the solutions. 
The optional text is used to give hints or instructions to the student in the questions.
- Outside of the R code: Use `<!--#!solutionbegin-->` and `<!--#!solutionend-->` to mark the solution part of the document.