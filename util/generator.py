import os
import pickle
import random
import re
import time


DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Generative template paths
LOCATIONS = os.path.join(DIR_PATH,
                         "problem_generation/locations.txt")
NAMES = os.path.join(DIR_PATH,
                     "problem_generation/names.txt")
OBJECTS = os.path.join(DIR_PATH,
                       "problem_generation/objects.txt")
ADDITION = os.path.join(DIR_PATH,
                        "problem_generation/addition.txt")
SUBTRACTION = os.path.join(DIR_PATH,
                           "problem_generation/subtraction.txt")
MULTIPLICATION = os.path.join(DIR_PATH,
                              "problem_generation/multiplication.txt")
DIVISION = os.path.join(DIR_PATH,
                        "problem_generation/division.txt")
COMPLEX = os.path.join(DIR_PATH,
                       "problem_generation/complex.txt")

PROBLEM_LIST = []


def get_random(min, max, float=False):
    if float:
        return random.uniform(min, max)
    else:
        return random.randint(min, max)


def generate(number, template="random", out_path="../data/generated/random_problems.pickle"):
    print(f"Generating {number} {template} problems...")
    if number > 5000:
        print(f"This could take a while...")

    start = time.time()

    if template == "addition":
        path = ADDITION
    elif template == "subtraction":
        path = SUBTRACTION
    elif template == "multiplication":
        path = MULTIPLICATION
    elif template == "division":
        path = DIVISION
    elif template == "complex":
        path = COMPLEX
    elif template == "random":
        path_list = [ADDITION, SUBTRACTION, DIVISION, MULTIPLICATION, COMPLEX]

    name_list = []
    location_list = []
    object_list = []
    template_list = []

    for i in range(number):
        num1 = get_random(0, 10)
        num2 = get_random(0, 10)

        u100_1 = get_random(0, 75)
        u100_2 = get_random(0, 75)

        float1 = "%.3f" % get_random(0, 100, float=True)
        float2 = "%.3f" % get_random(0, 100, float=True)

        dollar1 = "%.2f" % get_random(0, 10, float=True)
        dollar2 = "%.2f" % get_random(0, 10, float=True)
        dollar3 = "%.2f" % get_random(0, 300, float=True)

        # D3 in the templates must be the largest dollar amount
        while float(dollar3) < float(dollar1) or float(dollar3) < float(dollar2):
            dollar3 = "%.2f" % get_random(0, 300, float=True)

        # Determine the type of problem if randomly selected
        if template == "random":
            index = random.randint(0, len(path_list) - 1)
            path = path_list[index]

        with open(NAMES, 'r') as fh:
            lines = fh.readlines()

            for line in lines:
                name_list.append(line)

        with open(LOCATIONS, 'r') as fh:
            lines = fh.readlines()

            for line in lines:
                location_list.append(line)

        with open(OBJECTS, 'r') as fh:
            lines = fh.readlines()

            for line in lines:
                object_list.append(line)

        with open(path, 'r') as fh:
            lines = fh.readlines()

            if template == "random":
                index = random.randint(0, len(lines) - 1)
                sp = lines[index].split(', ')
                template_list.append((sp[0], sp[1]))
            else:
                for line in lines:
                    sp = line.split(', ')
                    template_list.append((sp[0], sp[1]))

        te = template_list[random.randint(0, len(template_list) - 1)]
        question = list(te)[0]
        equation = list(te)[1]

        # Choose the names so that they are different
        name1 = name_list[random.randint(0, len(name_list) - 1)]
        name2 = name_list[random.randint(0, len(name_list) - 1)]
        while name1 == name2:
            name2 = name_list[random.randint(0, len(name_list) - 1)]

        # Choose the objects so that they are different
        object1 = object_list[random.randint(0, len(object_list) - 1)]
        object2 = object_list[random.randint(0, len(object_list) - 1)]
        while object1 == object2:
            object2 = object_list[random.randint(0, len(object_list) - 1)]

        # Choose the locations so that they are different
        location1 = location_list[random.randint(0, len(location_list) - 1)]
        location2 = location_list[random.randint(0, len(location_list) - 1)]
        while location1 == location2:
            location2 = location_list[random.randint(
                0, len(location_list) - 1)]

        # Fill out the problem
        # Names
        question = re.sub("NA1", name1, question)
        question = re.sub("NA2", name2, question)
        # Under 100 integers
        question = re.sub("U100_1", str(u100_1), question)
        question = re.sub("U100_2", str(u100_2), question)
        # Dollar amounts
        question = re.sub("D1", dollar1, question)
        question = re.sub("D2", dollar2, question)
        question = re.sub("D3", dollar3, question)
        # Floating point numbers
        question = re.sub("F1", float1, question)
        question = re.sub("F2", float2, question)
        # Objects
        question = re.sub("OB1", object1, question)
        question = re.sub("OB2", object2, question)
        # Locations
        question = re.sub("LOC1", location1, question)
        question = re.sub("LOC2", location2, question)
        # Integers
        question = re.sub("N1", str(num1), question)
        question = re.sub("N2", str(num2), question)
        question = question.lower()

        # Fill out the equation
        # Integers
        equation = re.sub("N1", str(num1), equation)
        equation = re.sub("N2", str(num2), equation)
        # Floats
        equation = re.sub("F1", str(float1), equation)
        equation = re.sub("F2", str(float2), equation)
        # Under 100 integers
        equation = re.sub("U100_1", str(u100_1), equation)
        equation = re.sub("U100_2", str(u100_2), equation)
        # Dollar amounts
        equation = re.sub("D1", dollar1, equation)
        equation = re.sub("D2", dollar2, equation)
        equation = re.sub("D3", dollar3, equation)

        # Remove unwanted \n's
        question = re.sub(r"\n", '', question)
        equation = re.sub(r"\n", '', equation)

        PROBLEM_LIST.append([("question", question), ("equation", equation)])

    # Save as binary
    with open(os.path.join(DIR_PATH, out_path), "wb") as fh:
        pickle.dump(PROBLEM_LIST, fh)

    print(f"...done. Took {int(time.time() - start)}s")
    exit()


def read_data_from_file(path):
    with open(os.path.join(DIR_PATH, path), "rb") as fh:
        file_data = pickle.load(fh)

    for i in file_data:
        print(f"{i}\n")


if __name__ == "__main__":
    generate(500000,
             # template="subtraction",
             out_path="../data/generated/generated_data_v1.pickle")

    # read_data_from_file("../data/generated/generated_data_v1.pickle")
