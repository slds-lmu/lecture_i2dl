# This script converts all markdown templates to markdown questions and solutions.

from pathlib import Path

TEMPLATE_ROOT = 'templates'
QUESTION_ROOT = 'questions'
SOLUTIONS_ROOT = 'solutions'

SOLUTION_SKIP_EXEC_TAG = '#!TAG SKIPQUESTEXEC'
HW_BEGIN_TAG = '#!TAG HWBEGIN'
HW_END_TAG = '#!TAG HWEND'
MSG_TAG = '#!MSG'


def main():
    # Create target directories
    Path(QUESTION_ROOT).mkdir(exist_ok=True)
    Path(SOLUTIONS_ROOT).mkdir(exist_ok=True)

    # List all templates to convert
    md_templates = [file for file in Path(TEMPLATE_ROOT).iterdir()
                    if file.suffix == '.md']

    # Convert all template mds to question/solution mds
    for md_template in md_templates:
        is_in_hw_block = False

        question_file_name = md_template.name.replace('template', 'question')
        solution_file_name = md_template.name.replace('template', 'solution')
        md_question = Path(QUESTION_ROOT, question_file_name)
        md_solution = Path(SOLUTIONS_ROOT, solution_file_name)
        with open(md_template, 'r') as template_file, \
             open(md_question, 'w') as question_file, \
             open(md_solution, 'w') as solution_file:

            for line in template_file:
                # Check for no cell exec in question
                if SOLUTION_SKIP_EXEC_TAG in line:
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


if __name__ == '__main__':
    main()
