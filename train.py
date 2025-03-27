import torch
import torch.nn as nn

from data_prep import (
    all_categories,
    all_letters,
    category_lines,
    findFiles,
    letterToIndex,
    lineToTensor,
    n_categories,
    n_letters,
    readLines,
    unicodeToAscii,
)
from network import RNN

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
example = lineToTensor("abc")
hidden = rnn.initHidden()
output, next_hidden = rnn(example[0], hidden)
print(output, hidden)
