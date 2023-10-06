import numpy as np 
import matplotlib.pyplot as plt
import math

set_A = np.array([[0.24, 0.16, 0.32], [0.36, 0.48, 0.72]])
set_A = np.transpose(set_A)

set_B = np.array([[0.54, 0.66, 0.76, 0.41], [0.11, 0.39, 0.28, 0.53]])
set_B = np.transpose(set_B)

average_A = np.mean(set_A, axis=0)
print("Average of set A: ", average_A)
average_B = np.mean(set_B, axis=0)
print("Average of set B: ", average_B)

# Covariance matrix of set A
cov_A = np.cov(set_A, rowvar=False)
print("Covariance matrix of set A: \n", cov_A)

# Covariance matrix of set B
cov_B = np.cov(set_B, rowvar=False)
print("Covariance matrix of set B: \n", cov_B)

# Determinant of covariance matrix of set A
det_cov_A = np.linalg.det(cov_A)
print("Determinant of covariance matrix of set A: ", det_cov_A)

# Determinant of covariance matrix of set B
det_cov_B = np.linalg.det(cov_B)
print("Determinant of covariance matrix of set B: ", det_cov_B)

# Inverse of covariance matrix of set A
inv_cov_A = np.linalg.inv(cov_A)
print("Inverse of covariance matrix of set A: \n", inv_cov_A)

# Inverse of covariance matrix of set B
inv_cov_B = np.linalg.inv(cov_B)
print("Inverse of covariance matrix of set B: \n", inv_cov_B)

