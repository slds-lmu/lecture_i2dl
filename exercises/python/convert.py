# This script starts a process to convert one or all template markdowns to:
# --> Question and solution versions in markdown.
# --> Question and solution versions in executed notebooks.
# --> Question and solution versions in rendered pdfs.
#
# A specific file can be provided by '-f <file>' over CLI. Otherwise the whole
# template folder will be converted.

from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
import re
import shutil
import subprocess
from typing import List, Tuple, Callable

from tqdm import tqdm
import os
import json

TEMPLATE_ROOT = 'templates'
QUESTION_ROOT = 'questions'
SOLUTIONS_ROOT = 'solutions'

QUESTION_SKIP_EXEC_TAG = '#!TAG SKIPQUESTEXEC'
HW_BEGIN_TAG = '#!TAG HWBEGIN'
HW_END_TAG = '#!TAG HWEND'
MSG_TAG = '#!MSG'

MAX_WORKERS = 1


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
        elif not re.match(r'py-lab-\d{2}-template\.md', file_path.name):
            raise ValueError('"{}" does not follow naming convention '
                             '(py-lab-xx-template.md).'.format(file_path))
        elif not file_path.is_file():
            raise ValueError('"{}" is not a file.'.format(file_path))

        template_files = [file_path]

    # If no argument is given, list all template mds.
    else:
        template_pattern = TEMPLATE_ROOT + r'/py-labl-\d{2}-template\.md'
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


def create_ques_sol_mds(template_files: List[Path]) -> Tuple[List[Path], List[Path]]:
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


def convert_mds_to_ipynb(file_list: List[Path]) -> List[Path]:
    """
    Convert a list of markdowns to jupyter notebooks and execute them.

    :param file_list: Paths of markdowns to convert.
    :return: Paths to the obtained notebooks.
    """
    return _apply_multiproc(_convert_mds_to_ipynb, file_list, MAX_WORKERS)


def _convert_mds_to_ipynb(file_path: Path) -> Path:
    """Worker function for mds to ipynb conversion."""
    p = subprocess.run(['jupytext', '--to', 'ipynb', '--execute', file_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        print('CONVERSION FAILED FOR {}:'.format(file_path))
        print(p.stderr)

    return Path(str(file_path).replace('.md', '.ipynb'))


def remove_cell_artifacts(file_list: List[Path]) -> None:
    """
    Remove cell artifacts.

    Cells that are marked by the `%%script echo` trick for non-execution as conversion
    still have this printed & contained in the specific cell.
    This function deletes it to form a clean notebook.

    :param file_list: Paths of jupyter notebooks.
    """
    for file in file_list:
        subprocess.run(['gawk', '-i', 'inplace', '!/%%script echo /', file])


def convert_ipynb_to_pdf(file_list: List[Path]) -> List[Path]:
    """
    Convert a list of jupyter notebooks to pdfs.

    :param file_list: Paths of notebooks to convert to pdf.
    :return: Paths to the obtained pdfs.
    """
    pdf_list = _apply_multiproc(_convert_ipynb_to_pdf_worker, file_list, MAX_WORKERS)
    print("PDFs created")
    # Clean up unnecessary LaTex and auxiliary files.
    all_parent_dirs = set([file.parent for file in file_list])
    for parent_dir in all_parent_dirs:
        for file_path in parent_dir.iterdir():
            if file_path.suffix in ['.log', '.out', '.aux', '.tex']:
                file_path.unlink()
            elif file_path.is_dir() and '_files' in file_path.name:
                shutil.rmtree(file_path)

    return pdf_list


def _convert_ipynb_to_pdf_worker(file_path: Path) -> Path:
    """Worker function for ipynb to pdf conversion."""

    # Direct conversion to pdf with local images is extremely buggy.
    # A workaround is to convert to .tex and compile it manually with pdflatex
    p = subprocess.run(['jupyter', 'nbconvert', '--to', 'latex', file_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Extract the first heading from the notebook
    title = None
    try:
        with open(file_path, 'r') as f:
            notebook = json.load(f)
        
        # Look for the first markdown cell with a heading
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                # Look for a header that starts with # Lab
                match = re.search(r'# +Lab +\d+', source)
                if match:
                    title = match.group(0).strip("# ")
                    break
        title = f"Deep Learning {title} | Summer Term 2025"
    except Exception as e:
        print(f"Error extracting title: {e}")
    
    # If title was found, update the LaTeX file
    tex_file = str(file_path).replace('.ipynb', '.tex')
    if title:
        try:
            with open(tex_file, 'r') as f:
                tex_content = f.read()
            
            # Replace the title in the LaTeX file
            tex_content = re.sub(
                r'\\title\{.*?\}',
                rf'\\title{{{title}}}',
                tex_content
            )
            lecturers = "Emanuel Sommer, Prof. Dr. David Rügamer"
            tex_content = re.sub(
                r'(\\title\{.*?\})',
                rf'\1\n\\author{{{lecturers}}}',
                tex_content
            )
            
            with open(tex_file, 'w') as f:
                f.write(tex_content)
        except Exception as e:
            print(f"Error updating title in LaTeX: {e}")

    debug = True
    if not debug:
        p = subprocess.run(['pdflatex', '-output-directory', file_path.parent,
                            tex_file],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # for debugging purposes
        os.system((
            f'pdflatex -output-directory {file_path.parent} '
            f'{tex_file}'
        ))

    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        print('CONVERSION FAILED FOR {}:'.format(file_path))
        print(p.stderr)

    return Path(str(file_path).replace('.ipynb', '.pdf'))


def _apply_multiproc(
        func: Callable[[Path], Path], ls: List[Path], max_workers: int) -> List[Path]:
    """Utility function for multiprocessing a function applied to a list of args."""
    return_ls: List = []
    with Pool(processes=min(max_workers, len(ls))) as p:
        with tqdm(total=len(ls), leave=False) as pbar:
            for path in p.imap(func, ls):
                pbar.set_description('Processed {:30}'.format(path.name))
                pbar.update()
                return_ls.append(path)
    return return_ls


if __name__ == '__main__':
    main()
