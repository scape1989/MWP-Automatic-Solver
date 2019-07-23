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
WORD_NUMBERS = os.path.join(DIR_PATH,
                            "problem_generation/numberwords.txt")

WORD_PROBLEMS = False

WORD_PROBLEM_FILES = ["generated_data_v1.pickle",
                      "generated_word_data_v1.pickle"]

random.seed(1996)


def get_random(min, max, float=False):
    if float:
        return random.uniform(min, max)
    else:
        return random.randint(min, max)


def generate(number, template="random", out_path="../data/generated/random_problems.pickle", complex_problems=False):
    problem_list = []

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
        if complex_problems:
            path_list = [ADDITION, SUBTRACTION,
                         DIVISION, MULTIPLICATION, COMPLEX]
        else:
            path_list = [ADDITION, SUBTRACTION, DIVISION, MULTIPLICATION]

    opl = path_list
    name_list = []
    location_list = []
    object_list = []
    word_number_list = []
    word_operations = ["plus", "minus", "divide", "multiply"]

    with open(WORD_NUMBERS, 'r') as fh:
        lines = fh.readlines()

        for line in lines:
            word_number_list.append(line)

    progress_new = 0
    for i in range(number):
        progress_new = int((i / number) * 100)
        if not i == progress_new or i == 0:
            print(f"Progress: %d%%\r" % progress_new, end="")

        template_list = []

        # Determine the type of problem if randomly selected
        problem_type_split = int((number / len(opl)))
        if template == "random":
            if i % problem_type_split == 0 and not i < problem_type_split:
                path_list = path_list[1:]

        path = path_list[0]

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

        if not WORD_PROBLEMS:
            num1 = get_random(0, 100000)
            num2 = get_random(0, 100000)
            num3 = get_random(0, 100000)

            u100_1 = get_random(0, 100)
            u100_2 = get_random(0, 100)
            u100_3 = get_random(0, 100)
            u100_4 = get_random(0, 100)

            float1 = "%.3f" % get_random(0, 1000, float=True)
            float2 = "%.3f" % get_random(0, 1000, float=True)

            dollar1 = "%.2f" % get_random(0, 200, float=True)
            dollar2 = "%.2f" % get_random(0, 200, float=True)
            dollar3 = "%.2f" % get_random(0, 10000, float=True)

            # D3 in the templates must be the largest dollar amount
            while float(dollar3) < float(dollar1) or float(dollar3) < float(dollar2):
                dollar3 = "%.2f" % get_random(0, 10000, float=True)

        else:
            num1 = word_number_list[get_random(0, len(word_number_list) - 1)]
            num2 = word_number_list[get_random(0, len(word_number_list) - 1)]
            num3 = word_number_list[get_random(0, len(word_number_list) - 1)]

            u100_1 = word_number_list[get_random(1, 100)]
            u100_2 = word_number_list[get_random(1, 100)]
            u100_3 = word_number_list[get_random(1, 100)]
            u100_4 = word_number_list[get_random(1, 100)]

            float1 = word_number_list[get_random(0, len(word_number_list) - 1)]
            float2 = word_number_list[get_random(0, len(word_number_list) - 1)]

            d1i = get_random(0, len(word_number_list) - 1)
            d2i = get_random(0, len(word_number_list) - 1)
            d3i = get_random(0, len(word_number_list) - 1)

            # D3 in the templates must be the largest dollar amount
            # The index is mapable to the actual numeric values
            while d3i < d1i or d3i < d2i:
                d3i = get_random(0, len(word_number_list) - 1)

            dollar1 = word_number_list[d1i]
            dollar2 = word_number_list[d2i]
            dollar3 = word_number_list[d3i]

        # Choose the names so that they are different
        name1 = name_list[random.randint(0, len(name_list) - 1)]
        name2 = name_list[random.randint(0, len(name_list) - 1)]
        name3 = name_list[random.randint(0, len(name_list) - 1)]
        while name1 == name2 and name1 == name3 and name2 == name3:
            name2 = name_list[random.randint(0, len(name_list) - 1)]
            name3 = name_list[random.randint(0, len(name_list) - 1)]

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
        question = re.sub("NA3", name3, question)
        # Objects
        question = re.sub("OB1", object1, question)
        question = re.sub("OB2", object2, question)
        # Locations
        question = re.sub("LOC1", location1, question)
        question = re.sub("LOC2", location2, question)

        if WORD_PROBLEMS:
            # Need to replace operators befor the numbers are inserted
            # This avoids replacing hyphenated number words
            def pad(what):
                return " " + what + " "

            # Replace the operators with words
            equation = re.sub(r"\+", pad(word_operations[0]), equation)
            equation = re.sub(r"(-|\-)", pad(word_operations[1]), equation)
            equation = re.sub(r"\/", pad(word_operations[2]), equation)
            equation = re.sub(r"\*", pad(word_operations[3]), equation)
            equation = re.sub(r"3.14", 'PI', equation)

        # Integers
        question = re.sub("N1", str(num1), question)
        question = re.sub("N2", str(num2), question)
        question = re.sub("N3", str(num3), question)
        # Floating point numbers
        question = re.sub("F1", str(float1), question)
        question = re.sub("F2", str(float2), question)
        # Under 100 integers
        question = re.sub("U100_1", str(u100_1), question)
        question = re.sub("U100_2", str(u100_2), question)
        question = re.sub("U100_3", str(u100_3), question)
        question = re.sub("U100_4", str(u100_4), question)
        # Dollar amounts
        question = re.sub("D1", str(dollar1), question)
        question = re.sub("D2", str(dollar2), question)
        question = re.sub("D3", str(dollar3), question)

        question = question.lower()

        # Fill out the equation
        # Integers
        equation = re.sub("N1", str(num1), equation)
        equation = re.sub("N2", str(num2), equation)
        equation = re.sub("N3", str(num3), equation)
        # Floats
        equation = re.sub("F1", str(float1), equation)
        equation = re.sub("F2", str(float2), equation)
        # Under 100 integers
        equation = re.sub("U100_1", str(u100_1), equation)
        equation = re.sub("U100_2", str(u100_2), equation)
        equation = re.sub("U100_3", str(u100_3), equation)
        equation = re.sub("U100_4", str(u100_4), equation)
        # Dollar amounts
        equation = re.sub("D1", str(dollar1), equation)
        equation = re.sub("D2", str(dollar2), equation)
        equation = re.sub("D3", str(dollar3), equation)

        # Remove unwanted \n's
        question = re.sub(r"\n", '', question)
        equation = re.sub(r"\n", '', equation)

        problem_list.append([("question", question), ("equation", equation)])

    # Save as binary
    with open(os.path.join(DIR_PATH, out_path), "wb") as fh:
        pickle.dump(problem_list, fh)

    print(f"...done. Took {int(time.time() - start)}s")


if __name__ == "__main__":
    generate(50000,
             out_path="../generated/gen_v1.pickle",
             complex_problems=False)
