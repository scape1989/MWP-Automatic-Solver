from __future__ import absolute_import

import os
import sys
import json
import pickle
import re

from .classes.EquationConverter import EquationConverter


# Datasets used as of now:
#   -> AI2 Arithmetic Questions (Some still need to be manually copied)
#   -> Dolphin18k (Only 1-var)
#   -> MaWPS


DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Composite list of MWPs
PROBLEM_LIST = []

# The same list with all equations converted from infix to Polish notation
POLISH_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Reverse Polish notation
REVERSE_POLISH_CONVERTED_PROBLEM_LIST = []


def one_sentence_per_line_clean(text):
    # Replace . at end of sentence with a .\n
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)\s+?\.\s+?", ".\n",
                  text, flags=re.IGNORECASE)

    # Replace _?,_?_,or ? with ?\n
    text = re.sub(r"(\s+)?\?(\s+)?", "?\n",
                  text, flags=re.IGNORECASE)

    # Erradicate sentences starting with a space
    text = re.sub(r"^\s+", "",
                  text)

    return text


def filter_equation(text):
    # Remove unecessary characters in Dolphin18k data
    text = re.sub(r"(\r\n)?equ:\s+", ",",
                  text, flags=re.IGNORECASE)

    text = re.sub(r"unkn:(\s+)?\w+(\s+)?,", "",
                  text, flags=re.IGNORECASE)

    return text


def to_lower_case(text):
    # Convert strings to lowercase
    try:
        text = text.lower()
    except:
        pass

    return text


def transform_AI2():
    print("\nWorking on AI2 data...")

    # Get relative path to file
    path = os.path.join(DIR_PATH, "../datasets/AI2/AI2.txt")

    problem_list = []

    with open(path, "r") as fh:
        content = fh.readlines()

    iterator = iter(content)

    for i in range(len(content)):
        if i % 3 == 0 or i == 0:
            # The MWP
            question_text = one_sentence_per_line_clean(content[i].strip())

            problem = [("question", to_lower_case(question_text)),
                       ("answer", content[i + 1].strip()),
                       ("equation", to_lower_case(content[i + 2].strip()))]

            problem_list.append(problem)

            # Add the problem to the global list
            PROBLEM_LIST.append(problem)

            # Skip to the next MWP in data
            next(iterator)
            next(iterator)

    # print(problem_list)

    total_problems = int(len(content) / 3)

    print(f"-> Retrieved {len(problem_list)} / {total_problems} problems.")

    print("...done.\n")

    return "AI2"


def transform_Dolphin18k():
    print("\nWorking on Dolphin18k data...")

    path = os.path.join(DIR_PATH, "../datasets/Dolphin18k/Dolphin18k.json")

    problem_list = []

    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
        # A MWP
        problem = []

        has_all_data = True

        data = json_data[i]
        if "text" in data and "equations" in data and "ans" in data:
            for key, value in data.items():
                if key == "text" or key == "equations" or key == "ans":
                    if len(value) == 0 or (key == "ans" and not value.isdigit()):
                        has_all_data = False

                    if key == "text":
                        desired_key = "question"

                        value = one_sentence_per_line_clean(value)
                    elif key == "equations":
                        desired_key = "equation"

                        value = filter_equation(value)
                    elif key == "ans":
                        desired_key = "answer"

                    problem.append((desired_key, to_lower_case(value)))
        if has_all_data == True:
            problem_list.append(problem)

            # Add the problem to the global list
            PROBLEM_LIST.append(problem)

    # print(problem_list)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("..done.\n")

    return "Dolphin18k"


def transform_MaWPS():
    print("\nWorking on MaWPS data...")

    path = os.path.join(DIR_PATH, "../datasets/MaWPS/MaWPS.json")

    problem_list = []

    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
            # A MWP
        problem = []

        has_all_data = True

        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                        has_all_data = False

                    if key == "sQuestion":
                        desired_key = "question"

                        value = one_sentence_per_line_clean(value)
                    elif key == "lEquations":
                        desired_key = "equation"
                    elif key == "lSolutions":
                        desired_key = "answer"

                    if key == "lEquations" or key == "lSolutions":
                        problem.append((desired_key,
                                        to_lower_case(value[0])))
                    else:
                        problem.append((desired_key,
                                        to_lower_case(value)))

        if has_all_data == True:
            problem_list.append(problem)

            # Add the problem to the global list
            PROBLEM_LIST.append(problem)

    # print(problem_list)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("...done.\n")

    return "MaWPS"


def transform_all_datasets():
    total_datasets = []
    # Iteratively rework the data
    total_datasets.append(transform_AI2())
    total_datasets.append(transform_Dolphin18k())
    total_datasets.append(transform_MaWPS())

    return total_datasets


def read_data_from_file(path):
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    for i in file_data:
        print(f"{i}\n")


if __name__ == "__main__":
    print("Transforming all original datasets...")
    print("NOTE: Find resulting data binaries in data/")

    # Filter unecessary data and change infix notation to prefix notation
    # Desired data will be found in the datasets_prefix folder once completed
    total_filtered_datasets = transform_all_datasets()

    print(f"A total of {len(PROBLEM_LIST)} problems "
          + f"have been filtered from {len(total_filtered_datasets)} datasets.\n")

    print("Saving cleaned data to data.p file...")

    path = os.path.join(DIR_PATH, "../data/data.p")

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(PROBLEM_LIST, fh)

    print("...done.")

    # Run with option 1 to print the data after compilation
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        read_data_from_file(path)

    print("\nConverting found data to prefix notation...")

    for problem in PROBLEM_LIST:
        problem_dict = dict(problem)
        for key, value in problem_dict.items():
            if key == "equation":
                convert = EquationConverter()
                print(value)
                convert.eqset(value)
                convert.show_expression_tree()
