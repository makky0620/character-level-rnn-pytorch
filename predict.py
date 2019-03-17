# coding: utf-8

import torch
import sys
import string
from pprint import pprint

from util import load_data
from model import SimpleRNN
from train import line_to_tensor, letter_to_tensor, letter_to_index

all_letters = string.ascii_letters + ".,;'"

if __name__ == "__main__":

    all_categories, category_lines = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_categories = len(all_categories)
    n_letters = len(all_letters)
    n_hidden = 128

    model = SimpleRNN(n_letters, n_hidden, n_categories)
    model.load_state_dict(torch.load('model_data/model.pth'))

    input_name = sys.argv[1]
    input_tensor = line_to_tensor(input_name, n_letters)
    hidden = model.init_hidden()
    output, _ = model(input_tensor, hidden)

    topv, topi = output.topk(3)
    
    print(">{}".format(input_name))
    for i in range(3):
        value = topv[-1][0][i]
        language_i = topi[-1][0][i]

        print('{:.2f}: {}'.format(value, all_categories[language_i]))
    