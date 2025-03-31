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


# Random item from a list
def randomChoice(l):
    return l[torch.randint(0, len(l), (1,)).item()]


# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category):
    """category = 'Russian' -> tensor([[0, 1, 0, 0]])"""
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# ``LongTensor`` of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor
