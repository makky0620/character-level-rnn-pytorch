# coding: utf-8

import glob
import os
from pprint import pprint

def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return lines

def load_data():
    all_categories = []
    category_lines = {}
    
    for filename in glob.glob('../assets/data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return all_categories, category_lines


if __name__ == "__main__":
    all_categories, category_lines = load_data()