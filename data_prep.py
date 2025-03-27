import glob
import os
import string
import unicodedata

import torch

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1


def findFiles(path):
    return glob.glob(path)


# 1 - Convert names from UNICODE to ASCII
def unicodeToAscii(s):
    """Ślusàrski -> Slusarski"""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


def readLines(filename):
    with open(filename, encoding="utf-8") as f:
        return [unicodeToAscii(line.strip()) for line in f]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


category_lines = {}
all_categories = []
for filename in findFiles("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError("Data not found")


# print("# categories:", n_categories, all_categories)

# print(lineToTensor("abc")[0])
# hidden = torch.zeros(1, 128)
# print(hidden)
# print(torch.cat([lineToTensor("abc")[0], hidden]))
