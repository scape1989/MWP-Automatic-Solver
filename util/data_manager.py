import os
import sys
import json
import pickle
import re

# Dataset goals:
#   1. Each MWP must be one sentence per line.
#   2. Each original equation will be rewritten in Polish (prefix) notation
#       - This ensures parenthesis are not necessarily learned
#       - The end result will be recalled in infix notation for comparison to other's work


# Datasets used as of now:
#   -> AI2 Arithmetic Questions
#   -> Dolphin18k (Only 1-var)
#   -> MaWPS


DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Composite list of MWPs
PROBLEM_LIST = []


def one_sentence_per_line_clean(text):
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)\.\s+?", ".\n",
                  text, flags=re.IGNORECASE)

    text = re.sub(r"^\s+", "",
                  text)

    return text


def filter_equation(text):
    text = re.sub(r"(\r\n)?equ:\s+", ",",
                  text, flags=re.IGNORECASE)

    return text


def add_unkn_if_not_in_data(text):
    # text = re.sub(r"X=", "unkn: x,x=", text)

    return text


def to_lower_case(text):
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
            problem = [("question", content[i].strip()),
                       ("answer", content[i + 1].strip()),
                       ("equation", f"unkn: x,{content[i + 2].strip().lower()}")]

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

                    if key == "lEquations":
                        problem.append((desired_key,
                                        to_lower_case(f"unkn: x,{value[0]}")))
                    elif key == "lSolutions":
                        problem.append(
                            (desired_key, to_lower_case(value[0])))
                    else:
                        problem.append((desired_key, to_lower_case(value)))

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


def read_data(path):
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    for i in file_data:
        print(f"{i}\n")


def main():
    print("Transforming all original datasets...")
    print("NOTE: Find resulting data binaries in data/")

    # Filter unecessary data and change infix notation to prefix notation
    # Desired data will be found in the datasets_prefix folder once completed
    total_filtered_datasets = transform_all_datasets()

    print(f"A total of {len(PROBLEM_LIST)} problems "
          + f"have been filtered from {len(total_filtered_datasets)} datasets.\n")

    print("Saving clean data...")

    path = os.path.join(DIR_PATH, "../data/data.p")

    # Save as binary
    with open(path, "wb") as fh:
        pickle.dump(PROBLEM_LIST, fh)

    print("...done.")

    if len(sys.argv) > 1 and sys.argv[1] == "1":
        read_data(path)


if __name__ == "__main__":
    main()
