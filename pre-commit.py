#!/usr/bin/env python
"""
Run the following commands to set up the hook

    ln -s $(pwd)/pre-commit.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

"""
from __future__ import with_statement, print_function
import os
import re
import subprocess
import copy
import json

ANSWER_START_INDICATOR = "# ANSWERS"
ANSWER_END_INDICATOR = "# END ANSWERS"

EXERCISE_NB_REGEX = "^[AM]+\s+notebooks/lectures/.*/exercises.ipynb"

HTML_PREVIEW_NAME = "preview.html"


def system(*args, **kwargs):
    """
    Make a system call
    """
    kwargs.setdefault('stdout', subprocess.PIPE)
    proc = subprocess.Popen(args, **kwargs)
    out, err = proc.communicate()
    return out


def generate_answer_cell(source):
    """
    Strip out the ANSWER_START and ANSWER_END indicators
    to generate the answer notebook
    """
    return [
        line for line in source
        if line.strip() not in (ANSWER_START_INDICATOR, ANSWER_END_INDICATOR)
    ]


def generate_question_source(nb_file, source):
    """
    Remove everything wrapped in an ANSWER_START
    or ANSWER_END from the exercises to store in
    the questions notebook
    """

    unterminated_start = False

    question_source = []

    for line in source:
        stripped_line = line.strip()

        if stripped_line == ANSWER_START_INDICATOR:
            if unterminated_start:
                raise Exception(
                    "Two {} in a row in {}"
                    .format(
                        ANSWER_START_INDICATOR,
                        nb_file
                    )
                )

            unterminated_start = True

        elif stripped_line == ANSWER_END_INDICATOR:
            if not unterminated_start:
                raise Exception(
                    "{} without corresponding {} in {}"
                    .format(
                        ANSWER_END_INDICATOR,
                        ANSWER_START_INDICATOR,
                        nb_file
                    )
                )

            unterminated_start = False

        else:
            if not unterminated_start:
                question_source.append(line)

    if unterminated_start:
        raise Exception(
            "{} without corresponding {} in {}"
            .format(
                ANSWER_START_INDICATOR,
                ANSWER_END_INDICATOR,
                nb_file
            )
        )

    return question_source


def generate_notebooks(exercise_nb_path):
    """
    Takes in the exercise notebook and generates
    the question and answer notebooks
    """

    with open(exercise_nb_path) as exercise_nb_file:
        exercise_nb = json.load(exercise_nb_file)

        # copy the notebooks
        questions = copy.deepcopy(exercise_nb)
        answers = copy.deepcopy(exercise_nb)

        # generate the new cells for the questions
        # and answers notebooks
        questions["cells"] = []
        answers["cells"] = []

        for cell in exercise_nb["cells"]:
            source = cell["source"]

            question_cell = copy.copy(cell)
            question_cell["source"] = generate_question_source(
                exercise_nb_path,
                source
            )
            # ignore the outputs cells for questions
            question_cell["outputs"] = []
            questions["cells"].append(question_cell)

            answer_cell = copy.copy(cell)
            answer_cell["source"] = generate_answer_cell(source)
            answers["cells"].append(answer_cell)

        return questions, answers


def create_directories(nb_file):
    """
    Creates the directories to store the
    nbs if they don't exist
    """
    notebook_dir = os.path.dirname(nb_file)
    questions_dir = os.path.join(notebook_dir, "questions")
    answers_dir = os.path.join(notebook_dir, "answers")

    if not os.path.exists(questions_dir):
        os.makedirs(questions_dir)

    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)

    return questions_dir, answers_dir


def write_notebook(nb, nb_dir):
    """
    Writes the notebook to the corresponding
    directory
    """
    nb_path = os.path.join(
        nb_dir,
        "notebook.ipynb"
    )

    with open(nb_path, "w") as nb_file:
        json.dump(nb, nb_file)


def get_nb_path(nb_dir):
    """
    Helper function to get the notebook
    path from the directory
    """
    return os.path.join(
        nb_dir,
        "notebook.ipynb"
    )


def generate_preview(nb_dir):
    """
    Converts the notebooks to html files
    then moves the files to give them the correct name
    """
    nb_path = get_nb_path(nb_dir)

    # create the corresponding preview
    command = "jupyter nbconvert --to html {} --output {}".format(
        nb_path,
        HTML_PREVIEW_NAME
    )
    system(*command.split())


def run_notebook(nb_dir):
    """
    Executes all of the cells in a notebook
    """
    nb_path = get_nb_path(nb_dir)

    command = "jupyter nbconvert --to notebook --execute {} --output {}" \
        .format(
            nb_path,
            "notebook.ipynb"
        )
    system(*command.split())


def process_exercise_nb(exercise_nb_path):
    """
    For an individual notebook, this function
        1. creates the question and answers notebooks
        2. creates directories to store them
        3. writes the notebooks to those directories
        4. generates preview files for those notebooks
        5. runs the answer notebook to output all cells
    """
    # create the question and answer notebooks
    questions_nb, answers_nb = generate_notebooks(exercise_nb_path)

    # create directories for the questions and answers
    questions_dir, answers_dir = create_directories(exercise_nb_path)

    # write the notebooks to the directories
    write_notebook(
        questions_nb,
        questions_dir
    )
    write_notebook(
        answers_nb,
        answers_dir
    )

    # execute the answer notebook
    run_notebook(answers_dir)

    # generate previews in the two directories
    generate_preview(questions_dir)
    generate_preview(answers_dir)


def find_notebooks_to_process():
    modified_exercises = re.compile(
        EXERCISE_NB_REGEX,
        re.MULTILINE
    )
    all_git_activities = system('git', 'status', '--porcelain') \
        .decode("utf-8")

    # find the git activities associated with the exercises
    modified_exercises = modified_exercises.findall(all_git_activities)

    # split out the file names
    return list(set([nb[3:] for nb in modified_exercises]))


def main():
    """
    Processes all of the exercise notebooks that
    we've updated with this commit
    """

    # process the notebooks
    for nb_file in find_notebooks_to_process():
        print ("Processing {}...".format(nb_file))
        process_exercise_nb(nb_file)


if __name__ == '__main__':
    main()
