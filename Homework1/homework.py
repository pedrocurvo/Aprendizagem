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
data = data.drop(data[data['y1'] <= 0.4].index)


for i in range(1, 5):
    information_gain = calculate_information_gain(data[:], f'y{i}', 'y_out')
    print(f'Information Gain for y{i}: {information_gain} \n')

data2 = pd.DataFrame({'y1_spearman' : [3, 2, 1, 5, 4, 10, 12, 11, 7, 9, 6, 8],
'y2_spearman' : [8, 11, 3.5, 3.5, 3.5, 11, 3.5, 11, 8, 3.5, 8, 3.5]})

corr_matrix = data2.corr(method='pearson')
print(corr_matrix)
# Calculate Spearman's Rank Correlation Coefficient

import matplotlib.pyplot as plt


data = pd.DataFrame(
    {'y1': [0.24, 0.06, 0.04, 0.36, 0.32, 0.68, 0.90, 0.76, 0.46, 0.62, 0.44, 0.52],
    'y2': [1, 2, 0, 0, 0, 2, 0, 2, 1, 0, 1, 0],
    'y3': [1, 0, 0, 2, 0, 2, 1, 2, 1, 0, 2, 2],
    'y4': [0, 0, 0, 1, 2, 1, 2, 0, 1, 1, 2, 0],
    'y_out': ['A', 'B', 'B', 'C', 'C', 'A', 'A', 'A', 'B', 'B', 'C', 'C']
})
# Split the data by class
class_A = data[data['y_out'] == 'A']
class_B = data[data['y_out'] == 'B']
class_C = data[data['y_out'] == 'C']

# Create a figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the histograms for each class
ax.hist(class_A['y1'], bins=5, range=(0, 1), density=True, alpha=0.5, label='Class A')
ax.hist(class_B['y1'], bins=5, range=(0, 1), density=True, alpha=0.5, label='Class B')
ax.hist(class_C['y1'], bins=5, range=(0, 1), density=True, alpha=0.5, label='Class C')

# Add a legend and axis labels
ax.legend()
ax.set_xlabel('y1')
ax.set_ylabel('Relative frequency')

plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate example data
y_out = np.array(['A', 'B', 'B', 'C', 'C', 'A', 'A', 'A', 'B', 'B', 'C', 'C'])
y_pred = np.array(['A', 'B', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'B', 'C', 'C'])

# Define the class labels
class_labels = ['A', 'B', 'C']

# Calculate the confusion matrix
cm = confusion_matrix(y_out, y_pred, labels=class_labels)

# Convert the confusion matrix to a DataFrame and transpose it
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels).T

# Create a heatmap of the confusion matrix
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')

# Add labels to the x and y axes
plt.xlabel('Reais')
plt.ylabel('Previstos')

plt.show()
plt.savefig('images/confusion_matrix.png')
