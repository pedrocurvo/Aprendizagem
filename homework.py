import numpy as np
from math import log2

def calculate_entropy(data):
    unique_labels, label_counts = np.unique(data, return_counts=True)
    probabilities = label_counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(data, attribute, target):
    total_entropy = calculate_entropy(data[target])
    
    unique_values = np.unique(data[attribute])
    weighted_entropy = 0
    
    for value in unique_values:
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset[target])
        print(f'Entropy for {attribute} = {value}: {subset_entropy}')
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    print(f'Total_Entropy {total_entropy}, Weighted_Entropy {weighted_entropy}')
    return information_gain

import pandas as pd

data = pd.DataFrame(
    {'y1': [0.24, 0.06, 0.04, 0.36, 0.32, 0.68, 0.90, 0.76, 0.46, 0.62, 0.44, 0.52],
    'y2': [1, 2, 0, 0, 0, 2, 0, 2, 1, 0, 1, 0],
    'y3': [1, 0, 0, 2, 0, 2, 1, 2, 1, 0, 2, 2],
    'y4': [0, 0, 0, 1, 2, 1, 2, 0, 1, 1, 2, 0],
    'y_out': ['A', 'B', 'B', 'C', 'C', 'A', 'A', 'A', 'B', 'B', 'C', 'C']
})

for i in range(1, 5):
    information_gain = calculate_information_gain(data, f'y{i}', 'y_out')
    print(f'Information Gain for y{i}: {information_gain} \n')
