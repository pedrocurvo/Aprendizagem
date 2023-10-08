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
cov_A = np.cov(set_A, rowvar=False, bias=True)
print("Covariance matrix of set A: \n", cov_A)

# Covariance matrix of set B
cov_B = np.cov(set_B, rowvar=False, bias=True)
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

# Porbability of x_8 and x_9
x_8 = np.array([0.38, 0.52])
prob_x_8_A = (1 / (2 * math.pi * math.sqrt(det_cov_A))) * math.exp(-0.5 * np.dot(np.dot((x_8 - average_A), inv_cov_A), np.transpose(x_8 - average_A)))
prob_x_8_B = (1 / (2 * math.pi * math.sqrt(det_cov_B))) * math.exp(-0.5 * np.dot(np.dot((x_8 - average_B), inv_cov_B), np.transpose(x_8 - average_B)))
print("Probability of x_8 in gaussian A: ", prob_x_8_A)
print("Probability of x_8 in gaussian B: ", prob_x_8_B)

p_A = 3 / 7
p_B = 4 / 7
p_0_1_A = 1 / 3
p_0_1_B = 1 / 4
p_0_A = 1 / 3
p_0_B = 1 / 4
pA = p_A * p_0_1_A * p_0_A * prob_x_8_A
pB = p_B * p_0_1_B * p_0_B * prob_x_8_B
print("Probability of x_8 in set A: ", pA)
print("Probability of x_8 in set B: ", pB)
print("Probability of x_8 in set A under ML", pA / p_A)

# Porbability of x_9
x_9 = np.array([0.42, 0.59])
prob_x_9_A = (1 / (2 * math.pi * math.sqrt(det_cov_A))) * math.exp(-0.5 * np.dot(np.dot((x_9 - average_A), inv_cov_A), np.transpose(x_9 - average_A)))
prob_x_9_B = (1 / (2 * math.pi * math.sqrt(det_cov_B))) * math.exp(-0.5 * np.dot(np.dot((x_9 - average_B), inv_cov_B), np.transpose(x_9 - average_B)))
print("Probability of x_9 in gaussian A: ", prob_x_9_A)
print("Probability of x_9 in gaussian B: ", prob_x_9_B)



p_A = 3 / 7
p_B = 4 / 7
p_0_1_A = 1 / 3
p_0_1_B = 1 / 4
p_0_A = 1 / 3
p_0_B = 1 / 2
pA = p_A * p_0_1_A * p_0_A * prob_x_9_A
pB = p_B * p_0_1_B * p_0_B * prob_x_9_B
print("Probability of x_9 in set A: ", pA)
print("Probability of x_9 in set B: ", pB)

print("Probability of x_9 in set A under ML", pA / p_A)

