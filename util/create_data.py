from __future__ import absolute_import

import os
import sys
import json
import pickle
import re

from classes.EquationConverter import EquationConverter


# Datasets used as of now:
#   -> AI2 Arithmetic Questions (Some still need to be manually copied)
#   -> Dolphin18k (Only 1-var)
#   -> MaWPS

START_TOKEN = '<start>'

END_TOKEN = '<end>'

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Composite list of MWPs
PROBLEM_LIST = []

# The same list with all equations converted from infix to cleaned infix
CLEAN_INFIX_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Polish notation
POLISH_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Reverse Polish notation
REVERSE_POLISH_CONVERTED_PROBLEM_LIST = []

DATA_STATS = os.path.join(DIR_PATH,
                          "../data/statistics.txt")


def one_sentence_per_line_clean(text):
    # Replace . with _._
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)

    # Replace ?
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ",
                  text, flags=re.IGNORECASE)

    # Erradicate sentences starting with a space
    text = re.sub(r"^\s+", "", text)

    text = text.replace('\n', ' ')

    text = text.replace('%', ' percent')

    text = text.replace('$', ' $ ')

    text = text.replace(r"\s+", ' ')

    # text = f"{START_TOKEN} {text} {END_TOKEN}"

    text = re.sub(r"  ", " ", text)

    return text


def filter_dolphin_equation(text):
    # Remove unecessary characters in Dolphin18k data
    text = re.sub(r"(\r\n)?equ:\s+", ",",
                  text, flags=re.IGNORECASE)

    text = re.sub(r"unkn:(\s+)?\w+(\s+)?,", "",
                  text, flags=re.IGNORECASE)

    # Conditions for problem
    if len(text.split(',')) > 1  \
            or not re.match(r"(^([A-Z]|[a-z])+(\s+)?\=.*|.*(\s+)?\=([A-Z]|[a-z])+$)", text) \
            or re.match(r".*(!|\_).*", text) \
            or len(text) == 0:
        return False

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
    path = os.path.join(DIR_PATH, "../data/datasets/AI2/AI2.txt")

    problem_list = []

    with open(path, "r") as fh:
        content = fh.readlines()

    iterator = iter(content)

    for i in range(len(content)):
        if i % 3 == 0 or i == 0:
            # The MWP
            question_text = one_sentence_per_line_clean(content[i].strip())

            problem = [("question", to_lower_case(question_text)),
                       ("equation", to_lower_case(content[i + 2].strip())),
                       ("answer", content[i + 1].strip())]

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

    path = os.path.join(
        DIR_PATH, "../data/datasets/Dolphin18k/Dolphin18k.json")

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

                        value = filter_dolphin_equation(value)

                        if value == False:
                            has_all_data = False
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

    path = os.path.join(DIR_PATH, "../data/datasets/MaWPS/MaWPS.json")

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

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lEquations":
                        desired_key = "equation"

                        # print(value)
                        value = value[0]

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lSolutions":
                        desired_key = "answer"

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
    print("NOTE: Find resulting data binaries in the data folder.")

    # Filter unecessary data and change infix notation to prefix notation
    # Desired data will be found in the datasets_prefix folder once completed
    total_filtered_datasets = transform_all_datasets()

    print(f"A total of {len(PROBLEM_LIST)} problems "
          + f"have been filtered from {len(total_filtered_datasets)} datasets.\n")

    print("Saving cleaned data to original_data.pickle file...")

    path = os.path.join(DIR_PATH, "../data/original_data.pickle")

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(PROBLEM_LIST, fh)

    print("...done.")

    print("\n\nConverting found data to cleaned infix notation...")

    for problem in PROBLEM_LIST:
        problem_dict = dict(problem)

        prefix = []

        discard = False

        for key, value in problem_dict.items():
            if key == "equation":
                convert = EquationConverter()
                convert.eqset(value)
                prefix_value = convert.expr_as_infix()
                if re.match(r"[a-z] = .*\d+.*", prefix_value):
                    prefix.append((key, prefix_value))
                else:
                    discard = True
            else:
                prefix.append((key, value))

        if not discard:
            CLEAN_INFIX_CONVERTED_PROBLEM_LIST.append(prefix)

    print(f"A total of {len(CLEAN_INFIX_CONVERTED_PROBLEM_LIST)} infix "
          + "problems have been filtered.")

    print("\nSaving cleaned prefix data to infix_data.pickle file...")

    path = os.path.join(DIR_PATH, "../data/infix_data.pickle")

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(CLEAN_INFIX_CONVERTED_PROBLEM_LIST, fh)

    print("...done.")

    print("\n\nConverting found data to prefix notation...")

    for problem in PROBLEM_LIST:
        problem_dict = dict(problem)

        prefix = []

        discard = False

        for key, value in problem_dict.items():
            if key == "equation":
                convert = EquationConverter()
                convert.eqset(value)
                prefix_value = convert.expr_as_prefix()
                if re.match(r"[a-z] = .*\d+.*", prefix_value):
                    prefix.append((key, prefix_value))
                else:
                    discard = True
            else:
                prefix.append((key, value))

        if not discard:
            POLISH_CONVERTED_PROBLEM_LIST.append(prefix)

    print(f"A total of {len(POLISH_CONVERTED_PROBLEM_LIST)} prefix "
          + "problems have been filtered.")

    print("\nSaving cleaned prefix data to prefix_data.pickle file...")

    path = os.path.join(DIR_PATH, "../data/prefix_data.pickle")

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(POLISH_CONVERTED_PROBLEM_LIST, fh)

    print("...done.")

    print("\n\nConverting found data to postfix notation...")

    for problem in PROBLEM_LIST:
        problem_dict = dict(problem)

        postfix = []

        discard = False

        for key, value in problem_dict.items():
            if key == "equation":
                convert = EquationConverter()
                convert.eqset(value)
                postfix_value = convert.expr_as_postfix()
                if re.match(r"[a-z] = .*\d+.*", postfix_value):
                    postfix.append((key, postfix_value))
                else:
                    discard = True
            else:
                postfix.append((key, value))

        if not discard:
            REVERSE_POLISH_CONVERTED_PROBLEM_LIST.append(postfix)

    print(f"A total of {len(REVERSE_POLISH_CONVERTED_PROBLEM_LIST)} postfix "
          + "problems have been filtered.")

    print("\nSaving cleaned postfix data to postfix_data.pickle file...")

    path = os.path.join(DIR_PATH, "../data/postfix_data.pickle")

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(REVERSE_POLISH_CONVERTED_PROBLEM_LIST, fh)

    print("...done.")

    print("\nSaving all cleaned data to large_data.pickle file...")

    path = os.path.join(DIR_PATH, "../data/large_data.pickle")

    # Combine all representations
    total_data = []

    for p in PROBLEM_LIST:
        total_data.append(p)

    for p in CLEAN_INFIX_CONVERTED_PROBLEM_LIST:
        total_data.append(p)

    for p in POLISH_CONVERTED_PROBLEM_LIST:
        total_data.append(p)

    for p in REVERSE_POLISH_CONVERTED_PROBLEM_LIST:
        total_data.append(p)

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(total_data, fh)

    print("...done.")

    if os.path.isfile(DATA_STATS):
        os.remove(DATA_STATS)

    with open(DATA_STATS, "w") as fh:
        fh.write("Data file information. "
                 + "All of the binaries are described below.\n\n")
        fh.write("Original Data: ")
        fh.write("%d problems\n" % len(PROBLEM_LIST))
        fh.write("Clean Infix Data: ")
        fh.write("%d problems\n" % len(CLEAN_INFIX_CONVERTED_PROBLEM_LIST))
        fh.write("Prefix Data: ")
        fh.write("%d problems\n" % len(POLISH_CONVERTED_PROBLEM_LIST))
        fh.write("Postfix Data: ")
        fh.write("%d problems\n" % len(REVERSE_POLISH_CONVERTED_PROBLEM_LIST))
        fh.write("Large Data: ")
        fh.write("%d problems\n" % len(total_data))

    # path = os.path.join(DIR_PATH, "../data/infix_data.pickle")

    # read_data_from_file(path)
