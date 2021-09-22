# Introduction to Deep Learning - Python Labs

This directory contains PyTorch versions of the lab sessions.

## USER: How to get started

Every lab session is contained in one notebook.
You might need to install some requirement packages to be able to run everything.
For a full new setup it is recommended to create a dedicated environment for this
project.

```shell
python -m venv .venv
```

The freshly created environment needs to be activated.

```shell
source .venv/bin/activate
```

Now you can install the requirement dependencies over the `requirements.txt` file or
conveniently via the Makefile.

```shell
make requirements
```

To view the files in jupyter, start a notebook server in your environment.

```shell
jupyter notebook .
```

## DEVELOPER: How to get started

For better version control that notebooks are kept in markdown format.
Apart from the `requirements.txt` you will need to install the `jupytext` package over
pip.

You can directly edit the template`.md` files or convert them to `.ipynb` files with 
`make convert-template-md-to-ipynb`. Conversion back to `.md` is done via
`make convert-template-ipynb-to-md`.

To produce all files for questions and solutions (including `.md`, `.ipynb` and `.pdf`)
just call `make convert-full`.

Some tags in the template lead to different behavior in question and solution
versions:

- `#!TAG HWBEGIN` and `#!TAG HWEND`: Everything in between does two tags is treated as
homework for the student. The content in this block is only included in the solutions.
- `#!MSG <string>`:  This allows to include messages within the homework block and could
contain some hints for the student or assignments like "fill this function". The message
is only contained in the questions.
- `#!TAG SKIPQUESTEXEC`: Include this at the top of a cell, which should not be executed
in the questions and only in the solutions. This becomes handy, when there are gaps for
homework or yet undefined variables/functions that would throw an error.