import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_lang, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_lang = n_lang

        self.i2h = nn.Linear(input_size + hidden_size + n_lang, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, lang):
        combined = torch.cat((input, hidden, lang), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        print("input:", input.shape)
        print("hidden:", hidden.shape)
        print("combined:", combined.shape)
        print("output:", output.shape)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
