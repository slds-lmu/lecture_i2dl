import pyomo.environ as aml
import click
import csv
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import numpy as np


@click.command()
@click.argument('question-file', type=click.Path())
@click.option('--easy-qs-per-student', '-e', default=5)
@click.option('--hard-qs-per-student', '-h', default=5)
@click.option('--topics-per-student', '-t', default=8)
def main(question_file, easy_qs_per_student, hard_qs_per_student, topics_per_student):
    question_data = []
    with open(question_file) as f:
        for i, q in enumerate(csv.DictReader(f)):
            if q['topic'] != '?':
                question_data.append(q)
            else:
                print(f'Skipping question at row {i} because it does not have a topic')

    nh = sum(int(q['difficulty']) > 0.5 for q in question_data)
    print('Loaded', len(question_data), 'questions,', nh, 'easy and',
          len(question_data) - nh, 'hard ones')

    model = aml.ConcreteModel()

    model.questions = aml.RangeSet(0, len(question_data) - 1)
    model.students = aml.RangeSet(0, 25)
    model.topics = aml.RangeSet(0, len(set(int(q['topic']) for q in question_data)) - 1)

    # 1 if question is hard, 0 otherwise
    model.hard_questions = aml.Param(model.questions, initialize=lambda model, q:
        int(question_data[q]['difficulty']))

    # 1 if question q is in topic t
    model.question_topic = aml.Param(model.questions * model.topics, initialize=lambda model, q, t:
        int(question_data[q]['topic']) == t)

    # x(sq) = 1 <=> student s gets question q
    model.x = aml.Var(model.students * model.questions, domain=aml.Binary, initialize=0)

    # y(st) = 1 <=> student s has questions about topic t
    model.y = aml.Var(model.students * model.questions, domain=aml.Binary, initialize=0)

    # lower and upper bound for number students per question
    model.ql = aml.Var(domain=aml.NonNegativeIntegers, initialize=0)
    model.qh = aml.Var(domain=aml.NonNegativeIntegers, initialize=0)

    # each student gets five easy questions
    model.student_easy_questions = aml.Constraint(
        model.students, rule=lambda m, s: sum(
            model.x[s, q] * (1 - model.hard_questions[q])
            for q in m.questions
        ) == easy_qs_per_student,
    )

    # each student gets five hard questions
    model.student_hard_questions = aml.Constraint(
        model.students, rule=lambda m, s: sum(
            model.x[s, q] * model.hard_questions[q]
            for q in m.questions
        ) == hard_qs_per_student
    )

    # set lower and upper bounds for questions
    model.question_lower_bound = aml.Constraint(
        model.questions, rule=lambda m, q: sum(
            model.x[s, q] for s in m.students
        ) >= m.ql
    )
    model.question_upper_bound = aml.Constraint(
        model.questions, rule=lambda m, q: sum(
            model.x[s, q] for s in m.students
        ) <= m.qh
    )

    # set lower and upper bounds for topics
    model.th = aml.Constraint(
        model.students * model.topics, rule=lambda m, s, t: sum(
            model.x[s, q] for q in m.questions if m.question_topic[q, t] > 0.5
        ) >= model.y[s, t] - 0.5
    )
    model.tl = aml.Constraint(
        model.students * model.topics, rule=lambda m, s, t: (
            None, sum(
                model.x[s, q] - model.y[s, t]
                for q in m.questions if m.question_topic[q, t] > 0.5
            ), 0.5
        )
    )

    model.topic_count = aml.Constraint(
        model.students, rule=lambda m, s: sum(
            model.y[s, t] for t in m.topics
        ) == topics_per_student - 1
    )

    # objective
    model.objective = aml.Objective(
        rule=lambda m: m.qh - m.ql,
        sense=aml.minimize
    )

    solver = aml.SolverFactory('gurobi')
    res = solver.solve(model)

    # questions for each student
    for s in model.students:
        qs = [q for q in model.questions if aml.value(model.x[s, q]) > 0.5]
        ts = len(set(question_data[q]['topic'] for q in qs))
        print(f'Student {s} ({ts} topics):', ', '.join(map(str, qs)))

        eq = [question_data[q]['question'] for q in qs if aml.value(model.hard_questions[q]) > 0.5]
        hq = [question_data[q]['question'] for q in qs if aml.value(model.hard_questions[q]) < 0.5]
        #print('  Easy questions\n   ', '\n    '.join(eq))
        #print('  Hard questions\n   ', '\n    '.join(hq))

    # histogram of question assignment
    q_hist = {}
    for q in model.questions:
        n = sum(1 for s in model.students if aml.value(model.x[s, q]) > 0.5)
        if n not in q_hist:
            q_hist[n] = 0
        q_hist[n] += 1
    for k in sorted(q_hist.keys()):
        print(f'Number of questions assigned to {k} students:', q_hist[k])

if __name__ == '__main__':
    main()

