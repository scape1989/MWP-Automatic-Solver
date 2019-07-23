from __future__ import absolute_import

import pickle
import os
import sys

DIR_PATH = DIR_PATH = os.path.abspath(os.path.dirname(__file__))


def read_data_from_file(path):
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    else:
        limit = 25
    return file_data[:limit]


if __name__ == "__main__":
    print(read_data_from_file(os.path.join(
        DIR_PATH, f"../{sys.argv[1]}")))
