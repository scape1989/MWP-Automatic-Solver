class NumberTag():
    def __init__(self, sentence, equation):
        self.__original_sentence = sentence
        self.__original_equation = equation

        self.__number_map = self.__map_numbers(sentence, equation)

        self.__tagged_sentence = self.__number_map[0]
        self.__tagged_equation = self.__number_map[1]
        self.__lookup_table = self.__number_map[2]

    def __map_numbers(self, sentence, equation):
        # Replaces numbers in a sentence with keyed tags
        splitput = sentence.split()
        spliteq = equation.split()
        number_dict = {}
        lookup_dict = {}

        for i, word in enumerate(splitput):
            try:
                maybe_number = float(word)
                index = len(number_dict)

                if not word in number_dict:
                    number_dict[word] = f"<number{index}>"

                splitput[i] = number_dict[word]
            except:
                pass

        for i, word in enumerate(spliteq):
            try:
                if word in number_dict:
                    spliteq[i] = number_dict[word]
            except:
                pass

        for k, v in number_dict.items():
            lookup_dict[v] = k

        return " ".join(splitput), " ".join(spliteq), lookup_dict

    def get_originals(self):
        return self.__original_sentence, self.__original_equation

    def get_masked(self):
        return self.__tagged_sentence, self.__tagged_equation, self.__lookup_table


if __name__ == "__main__":
    problem, equation = "there are 14240 books in a library . they are arranged on shelves that hold 8 books each . how many shelves are in the library ?", "x = 14240 / 8"
    problem_tuple = NumberTag(problem, equation)

    print(problem_tuple.get_masked())

    print(problem_tuple.get_originals())
