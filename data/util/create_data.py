from __future__ import absolute_import

import os
import sys
import json
import pickle
import re
import random

from classes.EquationConverter import EquationConverter
from utils import to_binary

SEED = 420365

TRAIN_SPLIT = 0.1

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Composite list of MWPs
PROBLEM_LIST = []

# The same list with all equations converted from infix to cleaned infix
CLEAN_INFIX_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Polish notation
POLISH_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Reverse Polish notation
REVERSE_POLISH_CONVERTED_PROBLEM_LIST = []

GENERATED = []

AI2 = []
ILLINOIS = []
COMMONCORE = []
FREQUENCY = 5

PREFIX_TEST = []
POSTFIX_TEST = []
INFIX_TEST = []

DATA_STATS = os.path.join(DIR_PATH,
                          "../README.md")


def one_sentence_clean(text):
    # Replace . with _._
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)

    # Replace ?
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ",
                  text, flags=re.IGNORECASE)

    # Erradicate sentences starting with a space
    text = re.sub(r"^\s+", "", text)

    text = text.replace('\n', ' ')
    text = text.replace('\'', " '")

    text = text.replace('%', ' percent')

    text = text.replace('$', ' $ ')

    text = text.replace(r"\s+", ' ')

    text = re.sub(r"  ", " ", text)

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

    problem_list = []

    with open(os.path.join(DIR_PATH, "../datasets/AI2/questions.txt"), "r") as fh:
        content = fh.readlines()

    iterator = iter(content)

    for i in range(len(content)):
        if i % 3 == 0 or i == 0:
            # The MWP
            question_text = one_sentence_clean(content[i].strip())

            problem = [("question", to_lower_case(question_text)),
                       ("equation", to_lower_case(content[i + 2].strip())),
                       ("answer", content[i + 1].strip())]

            problem_list.append(problem)
            AI2.append(problem)

            # Skip to the next MWP in data
            next(iterator)
            next(iterator)

    total_problems = int(len(content) / 3)

    print(f"-> Retrieved {len(problem_list)} / {total_problems} problems.")

    print("...done.\n")

    return "AI2"


def transform_CommonCore():
    print("\nWorking on CommonCore data...")

    problem_list = []

    with open(os.path.join(DIR_PATH, "../datasets/CommonCore/questions.json"), encoding='utf-8-sig') as fh:
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

                        value = one_sentence_clean(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lEquations":
                        desired_key = "equation"

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
            COMMONCORE.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("...done.\n")

    return "CommonCore"


def transform_Illinois():
    print("\nWorking on Illinois data...")

    problem_list = []

    with open(os.path.join(DIR_PATH, "../datasets/Illinois/questions.json"), encoding='utf-8-sig') as fh:
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

                        value = one_sentence_clean(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lEquations":
                        desired_key = "equation"

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
            ILLINOIS.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("...done.\n")

    return "Illinois"


def transform_custom():
    print("\nWorking on generated data...")

    path = os.path.join(DIR_PATH, "../generated/gen_v1.pickle")

    problem_list = []

    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

        for problem in file_data:
            problem_list.append(problem)

            # Add the problem to the global list
            PROBLEM_LIST.append(problem)
            GENERATED.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(file_data)} problems.")

    print("...done.\n")

    return "Custom"


def transform_all_datasets():
    total_datasets = []

    # Iteratively rework the data
    total_datasets.append(transform_AI2())
    total_datasets.append(transform_CommonCore())
    total_datasets.append(transform_Illinois())
    total_datasets.append(transform_custom())

    return total_datasets


def convert_to(l, t):
    output = []

    for p in l:
        p_dict = dict(p)

        ol = []

        discard = False

        for k, v in p_dict.items():
            if k == "equation":
                convert = EquationConverter()
                convert.eqset(v)

                if t == "infix":
                    ov = convert.expr_as_infix()
                elif t == "prefix":
                    ov = convert.expr_as_prefix()
                elif t == "postfix":
                    ov = convert.expr_as_postfix()

                if re.match(r"[a-z] = .*\d+.*", ov):
                    ol.append((k, ov))
                else:
                    discard = True
            else:
                ol.append((k, v))

        if not discard:
            output.append(ol)

    return output


def duplicate_in_large_data(l):
    for i in range(FREQUENCY):
        for problem in l:
            PROBLEM_LIST.append(problem)


if __name__ == "__main__":
    print("Transforming all original datasets...")
    print("NOTE: Find resulting data binaries in the data folder.")

    total_filtered_datasets = transform_all_datasets()

    # Randomize
    random.shuffle(PROBLEM_LIST)

    random.shuffle(AI2)
    random.shuffle(COMMONCORE)
    random.shuffle(ILLINOIS)

    # Split
    AI2_TEST = AI2[:int(len(AI2) * TRAIN_SPLIT)]
    AI2 = AI2[int(len(AI2) * TRAIN_SPLIT):]

    COMMONCORE_TEST = COMMONCORE[:int(len(COMMONCORE) * TRAIN_SPLIT)]
    COMMONCORE = COMMONCORE[int(len(COMMONCORE) * TRAIN_SPLIT):]

    ILLINOIS_TEST = ILLINOIS[:int(len(ILLINOIS) * TRAIN_SPLIT)]
    ILLINOIS = ILLINOIS[int(len(ILLINOIS) * TRAIN_SPLIT):]

    # AI2 testing data
    test_pre_ai2 = convert_to(AI2_TEST, "prefix")
    test_pos_ai2 = convert_to(AI2_TEST, "postfix")
    test_inf_ai2 = convert_to(AI2_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_ai_prefix.pickle"),
              test_pre_ai2)
    to_binary(os.path.join(DIR_PATH, "../test_ai_postfix.pickle"),
              test_pos_ai2)
    to_binary(os.path.join(DIR_PATH, "../test_ai_infix.pickle"),
              test_inf_ai2)

    # AI2 training data
    pre_ai2 = convert_to(AI2, "prefix")
    pos_ai2 = convert_to(AI2, "postfix")
    inf_ai2 = convert_to(AI2, "infix")

    to_binary(os.path.join(DIR_PATH, "../ai_prefix.pickle"),
              pre_ai2)
    to_binary(os.path.join(DIR_PATH, "../ai_postfix.pickle"),
              pos_ai2)
    to_binary(os.path.join(DIR_PATH, "../ai_infix.pickle"),
              inf_ai2)

    # Common Core testing data
    test_pre_common = convert_to(COMMONCORE_TEST, "prefix")
    test_pos_common = convert_to(COMMONCORE_TEST, "postfix")
    test_inf_common = convert_to(COMMONCORE_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_cc_prefix.pickle"),
              test_pre_common)
    to_binary(os.path.join(DIR_PATH, "../test_cc_postfix.pickle"),
              test_pos_common)
    to_binary(os.path.join(DIR_PATH, "../test_cc_infix.pickle"),
              test_inf_common)

    # Common Core training data
    pre_common = convert_to(COMMONCORE, "prefix")
    pos_common = convert_to(COMMONCORE, "postfix")
    inf_common = convert_to(COMMONCORE, "infix")

    to_binary(os.path.join(DIR_PATH, "../cc_prefix.pickle"),
              pre_common)
    to_binary(os.path.join(DIR_PATH, "../cc_postfix.pickle"),
              pos_common)
    to_binary(os.path.join(DIR_PATH, "../cc_infix.pickle"),
              inf_common)

    # Common Core testing data
    test_pre_il = convert_to(ILLINOIS_TEST, "prefix")
    test_pos_il = convert_to(ILLINOIS_TEST, "postfix")
    test_inf_il = convert_to(ILLINOIS_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_il_prefix.pickle"),
              test_pre_il)
    to_binary(os.path.join(DIR_PATH, "../test_il_postfix.pickle"),
              test_pos_il)
    to_binary(os.path.join(DIR_PATH, "../test_il_infix.pickle"),
              test_inf_il)

    # Common Core testing data
    pre_il = convert_to(ILLINOIS, "prefix")
    pos_il = convert_to(ILLINOIS, "postfix")
    inf_il = convert_to(ILLINOIS, "infix")

    to_binary(os.path.join(DIR_PATH, "../il_prefix.pickle"),
              pre_il)
    to_binary(os.path.join(DIR_PATH, "../il_postfix.pickle"),
              pos_il)
    to_binary(os.path.join(DIR_PATH, "../il_infix.pickle"),
              inf_il)

    # Duplicate data in large training set 5 times
    duplicate_in_large_data(AI2)
    duplicate_in_large_data(COMMONCORE)
    duplicate_in_large_data(ILLINOIS)

    print(f"A total of {len(PROBLEM_LIST)} problems "
          + f"have been filtered from {len(total_filtered_datasets)} datasets.\n")

    print("\n\nConverting found data to cleaned infix notation...")

    CLEAN_INFIX_CONVERTED_PROBLEM_LIST = convert_to(PROBLEM_LIST, "infix")

    print(f"A total of {len(CLEAN_INFIX_CONVERTED_PROBLEM_LIST)} infix "
          + "problems have been filtered.")

    print("\nSaving infix data to train_infix.pickle file...")

    to_binary(os.path.join(DIR_PATH, "../train_infix.pickle"),
              CLEAN_INFIX_CONVERTED_PROBLEM_LIST)

    print("...done.")

    print("\n\nConverting found data to prefix notation...")

    POLISH_CONVERTED_PROBLEM_LIST = convert_to(PROBLEM_LIST, "prefix")

    print(f"A total of {len(POLISH_CONVERTED_PROBLEM_LIST)} prefix "
          + "problems have been filtered.")

    print("\nSaving prefix data to train_prefix.pickle file...")

    to_binary(os.path.join(DIR_PATH, "../train_prefix.pickle"),
              POLISH_CONVERTED_PROBLEM_LIST)

    print("...done.")

    print("\n\nConverting found data to postfix notation...")

    REVERSE_POLISH_CONVERTED_PROBLEM_LIST = convert_to(PROBLEM_LIST, "postfix")

    print(f"A total of {len(REVERSE_POLISH_CONVERTED_PROBLEM_LIST)} postfix "
          + "problems have been filtered.")

    print("\nSaving postfix data to train_postfix.pickle file...")

    to_binary(os.path.join(DIR_PATH, "../train_postfix.pickle"),
              REVERSE_POLISH_CONVERTED_PROBLEM_LIST)

    print("...done.")

    print("\nCreating a small debugging file...")

    # Combine all representations
    small_data = []

    for p in PROBLEM_LIST[:100]:
        small_data.append(p)

    to_binary(os.path.join(DIR_PATH, "../debug.pickle"), small_data)

    print("...done.")

    if os.path.isfile(DATA_STATS):
        os.remove(DATA_STATS)

    with open(DATA_STATS, "w") as fh:
        fh.write("Data file information. "
                 + "All of the binaries are described below.\n\n")
        fh.write(f"Testing Split: {TRAIN_SPLIT * 100}%\n\n")
        fh.write("Original: ")
        fh.write("%d problems\n" % len(PROBLEM_LIST))
        fh.write("Debugging Data: ")
        fh.write("%d problems\n" % len(small_data))
        fh.write("\nInfix Data: ")
        fh.write("%d problems\n" % len(CLEAN_INFIX_CONVERTED_PROBLEM_LIST))
        fh.write("Prefix Data: ")
        fh.write("%d problems\n" % len(POLISH_CONVERTED_PROBLEM_LIST))
        fh.write("Postfix Data: ")
        fh.write("%d problems\n" % len(REVERSE_POLISH_CONVERTED_PROBLEM_LIST))
        fh.write("\nAI2 Train: ")
        fh.write("%d problems\n" % len(AI2))
        fh.write("Common Core Train: ")
        fh.write("%d problems\n" % len(COMMONCORE))
        fh.write("Illinois Train: ")
        fh.write("%d problems\n" % len(ILLINOIS))
        fh.write("Generated MWPs (gen): ")
        fh.write("%d problems\n" % len(GENERATED))
        fh.write("\nAI2 Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_ai2))
        fh.write("AI2 Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_ai2))
        fh.write("AI2 Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_ai2))
        fh.write("Common Core Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_common))
        fh.write("Common Core Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_common))
        fh.write("Common Core Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_common))
        fh.write("Illinois Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_il))
        fh.write("Illinois Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_il))
        fh.write("Illinois Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_il))
