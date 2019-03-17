# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
from copy import deepcopy
from pprint import pprint

from util import load_data
from model import SimpleRNN

all_letters = string.ascii_letters + ".,;'"

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter, n_letters):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line, n_letters):
    tensor = torch.zeros(len(line), 1, n_letters)

    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_example(all_categories, category_lines, n_letters):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line, n_letters)
    return category, line, category_tensor, line_tensor

if __name__ == "__main__":

    all_categories, category_lines = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_categories = len(all_categories)
    n_letters = len(all_letters)
    n_hidden = 128


    model = SimpleRNN(n_letters, n_hidden, n_categories).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(1, 100000+1):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines, n_letters)
        category_tensor, line_tensor = category_tensor.to(device), line_tensor.to(device)

        hidden = model.init_hidden()
        hidden = hidden.to(device)
        model.zero_grad()

        output, hidden = model(line_tensor, hidden)
        loss = criterion(output[-1], category_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 5000 == 0:
            print("epoch {} loss {}".format(epoch, loss))
    
    torch.save(deepcopy(model).cpu().state_dict(), 'model_data/model.pth')