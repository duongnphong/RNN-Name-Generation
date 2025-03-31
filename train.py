import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data_prep import n_categories, n_letters, randomTrainingExample
from network import RNN


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = torch.Tensor([0])

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


rnn = RNN(n_letters, 128, n_categories, n_letters)

criterion = nn.NLLLoss()
learning_rate = 0.0005
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0  # Reset every ``plot_every`` ``iters``

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print(
            "%s (%d %d%%) %.4f" % (timeSince(start), iter, iter / n_iters * 100, loss)
        )

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

torch.save(rnn.state_dict(), "model.pth")
print("Model saved as model.pth")


plt.figure()
plt.plot(all_losses)
plt.savefig("loss.png")
