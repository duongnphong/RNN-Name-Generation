import torch

from data_prep import all_letters, categoryTensor, inputTensor, n_categories, n_letters
from network import RNN

n_hidden = 128
# Initialize the model
rnn = RNN(n_letters, n_hidden, n_categories, n_letters)

# Load the saved state dictionary
rnn.load_state_dict(torch.load("model.pth"))
rnn.eval()  # Set to evaluation mode

# Your sampling functions
max_length = 20


def sample(category, start_letter="A"):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()
        output_name = start_letter
        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name


def samples(category, start_letters="ABC"):
    for start_letter in start_letters:
        print(sample(category, start_letter))


# Generate names
print("Vietnamese names:")
samples("Vietnamese", "VNM")

print("\nGerman names:")
samples("German", "GER")

print("\nJapanese names:")
samples("Japanese", "JPN")

print("\nChinese names:")
samples("Chinese", "CHI")
