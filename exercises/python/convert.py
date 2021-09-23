# This script starts a process to convert one or all template markdowns to:
# --> Question and solution versions in markdown.
# --> Question and solution versions in executed notebooks.
# --> Question and solution versions in rendered pdfs.
#
# A specific file can be provided by '-f <file>' over CLI. Otherwise the whole
# template folder will be converted.

from argparse import ArgumentParser
from pathlib import Path
import re
import subprocess
from typing import List, Tuple

from tqdm import tqdm

TEMPLATE_ROOT = 'templates'
QUESTION_ROOT = 'questions'
SOLUTIONS_ROOT = 'solutions'

QUESTION_SKIP_EXEC_TAG = '#!TAG SKIPQUESTEXEC'
HW_BEGIN_TAG = '#!TAG HWBEGIN'
HW_END_TAG = '#!TAG HWEND'
MSG_TAG = '#!MSG'


def main():
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    args = parser.parse_args()

    ###################################################################################
    # CREATE FILE LISTS
    ###################################################################################

    # Check argument
    if args.file:
        file_path = Path(args.file)
        if str(file_path.parent) != TEMPLATE_ROOT:
            raise ValueError('"{}" does not point to the template subdirectory.'
                             .format(file_path))
        elif not re.match(r'\d{2}-lab-template\.md', file_path.name):
            raise ValueError('"{}" does not follow naming convention '
                             '(xx-lab-template.md).'.format(file_path))
        elif not file_path.is_file():
            raise ValueError('"{}" is not a file.'.format(file_path))

        template_files = [file_path]

    # If no argument is given, list all template mds.
    else:
        template_pattern = TEMPLATE_ROOT + r'/\d{2}-lab-template\.md'
        template_files = [entry for entry in Path(TEMPLATE_ROOT).iterdir()
                          if re.match(template_pattern, str(entry))]

    ###################################################################################
    # CREATE QUESTION/SOLUTION MDS
    ###################################################################################
    print('-- Started  -- Creating question & solutions markdowns.')
    question_mds, solution_mds = create_ques_sol_mds(template_files)
    print('-- Finished -- Creating question & solutions markdowns.')

    ###################################################################################
    # CONVERT MDS TO IPYNB AND EXECUTE
    ###################################################################################
    print('-- Started  -- Converting & executing question markdowns.')
    question_notebooks = convert_mds_to_ipynb(question_mds)
    remove_cell_artifacts(question_notebooks)
    print('-- Finished -- Converting & executing question markdowns.')

    print('-- Started  -- Converting & executing solution markdowns.')
    solution_notebooks = convert_mds_to_ipynb(solution_mds)
    print('-- Finished -- Converting & executing solution markdowns.')

    ###################################################################################
    # CONVERT IPYNB TO PDF
    ###################################################################################
    print('-- Started  -- Converting notebooks to pdfs.')
    convert_ipynb_to_pdf(question_notebooks + solution_notebooks)
    print('-- Finished -- Converting notebooks to pdfs.')


def create_ques_sol_mds(template_files: List) -> Tuple[List, List]:
    """
    Create question and solution markdowns.

    :param template_files: List of paths pointing to template files.
    :return: A tuple of two lists: Paths to question mds and paths to solution mds
    """
    # Create target directories
    Path(QUESTION_ROOT).mkdir(exist_ok=True)
    Path(SOLUTIONS_ROOT).mkdir(exist_ok=True)

    # Create file lists
    question_files: List = []
    solution_files: List = []

    # Convert all template mds to question/solution mds
    for md_template in tqdm(template_files, leave=False):
        is_in_hw_block = False

        question_file_name = md_template.name.replace('template', 'question')
        solution_file_name = md_template.name.replace('template', 'solution')

        md_question = Path(QUESTION_ROOT, question_file_name)
        md_solution = Path(SOLUTIONS_ROOT, solution_file_name)

        question_files.append(md_question)
        solution_files.append(md_solution)

        with open(md_template, 'r') as template_file, \
                open(md_question, 'w') as question_file, \
                open(md_solution, 'w') as solution_file:

            for line in template_file:
                # Check for no cell exec in question
                if QUESTION_SKIP_EXEC_TAG in line:
                    question_file.write('%%script echo \n')

                # Check for HW Tags
                elif HW_BEGIN_TAG in line:
                    is_in_hw_block = True
                elif HW_END_TAG in line:
                    is_in_hw_block = False

                # Write to files according to HW Tag
                elif is_in_hw_block and MSG_TAG in line:
                    line = '# ' + line.replace(MSG_TAG, '').strip() + '\n'
                    question_file.write(line)
                elif is_in_hw_block:
                    solution_file.write(line)
                else:
                    question_file.write(line)
                    solution_file.write(line)

    return question_files, solution_files


def convert_mds_to_ipynb(file_list: List) -> List:
    """
    Convert a list of markdowns to jupyter notebooks and execute them.

    :param file_list: Paths of markdowns to convert.
    :return: Paths to the obtained notebooks.
    """
    notebook_list: List = []

    pbar = tqdm(file_list, leave=False)
    for md_file in pbar:
        pbar.set_description('Processing {:30}'.format(str(md_file)))

        p = subprocess.run(['jupytext', '--to', 'ipynb', '--execute', md_file],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            p.check_returncode()
        except subprocess.CalledProcessError:
            print('CONVERSION FAILED FOR {}:'.format(md_file))
            print(p.stderr)

        notebook_list.append(str(md_file).replace('.md', '.ipynb'))

    return notebook_list


def remove_cell_artifacts(file_list: List) -> None:
    """
    Remove cell artifacts.

    Cells that are marked by the `%%script echo` trick for non-execution as conversion
    still have this printed & contained in the specific cell.
    This function deletes it to form a clean notebook.

    :param file_list: Paths of jupyter notebooks.
    """
    for file in file_list:
        subprocess.run(['gawk', '-i', 'inplace', '!/%%script echo /', file])


def convert_ipynb_to_pdf(file_list: List) -> List:
    """
    Convert a list of jupyter notebooks to pdfs.

    :param file_list: Paths of notebooks to convert to pdf.
    :return: Paths to the obtained pdfs.
    """
    pdf_list: List = []
    pbar = tqdm(file_list, leave=False)
    for notebook_file in pbar:
        pbar.set_description('Processing {:30}'.format(str(notebook_file)))

        # Direct conversion to pdf with local images is extremely buggy.
        # A workaround is to convert to .tex and compile it manually with pdflatex
        p = subprocess.run(['jupyter', 'nbconvert', '--to', 'latex', notebook_file],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        p = subprocess.run(['pdflatex', '-output-directory', Path(notebook_file).parent,
                            notebook_file.replace('.ipynb', '.tex')],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for file_path in Path(notebook_file).parent.iterdir():
            if file_path.suffix in ['.log', '.out', '.aux', '.tex']:
                file_path.unlink()

        try:
            p.check_returncode()
        except subprocess.CalledProcessError:
            print('CONVERSION FAILED FOR {}:'.format(notebook_file))
            print(p.stderr)

        pdf_list.append(str(notebook_file).replace('.ipynb', '.pdf'))

    return pdf_list


if __name__ == '__main__':
    main()
