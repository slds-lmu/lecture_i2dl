import os
import re
import sys


def convert_file(fname, questions_dest, solutions_dest):
  with open(fname) as fin:
    with open(questions_dest, 'w') as fqs:
      with open(solutions_dest, 'w') as fss:
        inside_question_block = is_homework = inside_answer_block = False

        for i, row in enumerate(fin):
          if row.strip().startswith('#!hwbegin') or row.strip().startswith('<!--#!solutionbegin-->'):
            if inside_question_block:
              print('Missing #!hwend, processing interrupted (currently at line %d)' % (i + 1))
              return

            print('    L%04d - ' % (i + 1), row.strip())

            if row.strip().startswith('<!--#!solutionbegin-->'):
                inside_answer_block = True
                inside_question_block = False
            else:
                inside_answer_block = False
                inside_question_block = True

            indent = len(row) - len(row.lstrip())
            row = row.replace('#!hwbegin', '#')
            row = row.replace('<!--#!solutionbegin-->', '')
            row = row.replace('\\n', '\n' + ' ' * indent)
            if not re.match(r'^\s*#\s*$', row):
                fqs.write(row)  # write only if commment not empty

            is_homework = True

          elif row.strip().startswith('#!hwend'):
            if not inside_question_block:
              print('Missing #!hwbegin, processing interrupted (currently at line %d)' % (i + 1))
            inside_question_block = False

          elif row.strip().startswith('<!--#!solutionend-->'):
            inside_question_block = inside_answer_block = False  # not much validation..

          elif row == '```\n' and inside_question_block:
            print('WARNING: Reached code block before #!hwend, will insert it for you (currently at line %d)' % (i + 1))
            inside_question_block = False
            fqs.write(row)

          elif not inside_question_block and not inside_answer_block:
            fqs.write(row)

            if row.startswith('knitr::opts_chunk$set'):
              row = row.replace('eval = FALSE', 'eval = TRUE')
            fss.write(row)
          else:
            fss.write(row)

  if not is_homework:
    print('Not an assignment')
    os.remove(questions_dest)
    os.remove(solutions_dest)


def main():
  if len(sys.argv) == 1:
    to_convert = [f for f in os.listdir('.') if f.endswith('-template.Rmd')]
  else:
    to_convert = sys.argv[1:]

  if not to_convert:
    print('WARNING: no R notebook found')

  for f in to_convert:
    print('Processing', f)
    if f.endswith('-template.Rmd'):
      questions_dest = 'generated/' + f.replace('-template.Rmd', '-questions.Rmd')
      solutions_dest = 'generated/' + f.replace('-template.Rmd', '-solutions.Rmd')
    else:
      questions_dest = 'generated/' + f.replace('.Rmd', '-questions.Rmd')
      solutions_dest = 'generated/' + f.replace('.Rmd', '-solutions.Rmd')

    convert_file(f, questions_dest, solutions_dest)


if __name__ == '__main__':
  main()
