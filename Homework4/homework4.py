import numpy as np
from scipy.stats import multivariate_normal, bernoulli

# Given observations
observations = np.array([[1, 0.6, 0.1], [0, -0.4, 0.8], [0, 0.2, 0.5], [1, 0.4, -0.1]])

# Initiazlization 
# E-Step (Calculate the posterior probability of each component given the observations,
#       that is, the probability of each point to belong to each cluster)

# We have two clusters 

# Cluster 1
pi1 = 0.5
p1 = 0.3 # p1 = P(y1 = 1)
mu1 = np.array([1, 1])
sigma1 = np.array([[2, 0.5], [0.5, 2]])

# Cluster 2
pi2 = 0.5
p2 = 0.7 # p2 = P(y1 = 1)
mu2 = np.array([0, 0])
sigma2 = np.array([[1.5, 1], [1, 1.5]])

# Given observations
observations = np.array([[1, 0.6, 0.1], [0, -0.4, 0.8], [0, 0.2, 0.5], [1, 0.4, -0.1]])

# Initialization
# We have two clusters 

# Cluster 1
pi1 = 0.5
p1 = 0.3 # p1 = P(y1 = 1)
mu1 = np.array([1, 1])
sigma1 = np.array([[2, 0.5], [0.5, 2]])

# Cluster 2
pi2 = 0.5
p2 = 0.7 # p2 = P(y1 = 1)
mu2 = np.array([0, 0])
sigma2 = np.array([[1.5, 1], [1, 1.5]])

# E-Step (Calculate the posterior probability of each component given the observations,
# that is, the probability of each point to belong to each cluster)

# Dicionary to store the responsabilities
responsabilities = {
    '1': [],
    '2': []
}

# For each observation
for i in range(len(observations)):
    print('\n')
    print(f'Observation {i+1}: {observations[i]}')
    # Calculate the Gaussian and Bernoulli probabilities for each cluster
    gaussian_1 = multivariate_normal.pdf(observations[i][1:], mean=mu1, cov=sigma1)
    print(f'Gaussian 1_{i+1}: {gaussian_1:.5f}')
    bernoulli_1 = round(bernoulli.pmf(observations[i][0], p1), 2)
    print(f'Bernoulli 1_{i+1}: {bernoulli_1:.5f}')
    gaussian_2 = multivariate_normal.pdf(observations[i][1:], mean=mu2, cov=sigma2)
    print(f'Gaussian 2_{i+1}: {gaussian_2:.5f}')
    bernoulli_2 = round(bernoulli.pmf(observations[i][0], p2), 5)
    print(f'Bernoulli 2_{i+1}: {bernoulli_2:.5f}')
    
    # Calculate the responsibilities for each cluster
    responsability_1 = gaussian_1 * bernoulli_1 * pi1
    responsability_2 = gaussian_2 * bernoulli_2 * pi2
    print(f'Responsability 1_{i+1}: {responsability_1:.5f}')
    print(f'Responsability 2_{i+1}: {responsability_2:.5f}')
    
    # Normalize the responsibilities
    sum_responsabilities = responsability_1 + responsability_2
    responsability_1 /= sum_responsabilities
    responsability_2 /= sum_responsabilities
    print(f'Normalized responsability 1_{i+1}: {responsability_1:.5f}')
    print(f'Normalized responsability 2_{i+1}: {responsability_2:.5f}')
    
    # Store the responsibilities
    responsabilities['1'].append(responsability_1)
    responsabilities['2'].append(responsability_2)
    

# M-Step (Update the parameters of the model, that is, the parameters of the clusters)
print('\n------------------------------M-Step-----------------------------\n')
mu1 = 0
for i in range(len(observations)):
    mu1 += responsabilities['1'][i] * observations[i][1:]
mu1 = mu1 / np.sum(responsabilities['1'])
print(f'mu1: {mu1}')
# print a string with mu1 in latex code:
latex_code = "mu1 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(mu1, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix}'
print(latex_code)

sigma1 = np.zeros((2, 2))
for i in range(len(observations)):
   array = np.expand_dims(observations[i][1:] - mu1, axis=1)
   multiplication = np.dot(array, array.T)
   sigma1 += responsabilities['1'][i] * multiplication
sigma1 = sigma1 / np.sum(responsabilities['1'])
# print sigma1 with 5 decimals
print(f'sigma1:\n ', np.round(sigma1, 5))
# print a string with sigma1 in latex code:
latex_code = "sigma1 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(sigma1, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix}'
print(latex_code)

p1 = 0
for i in range(len(observations)):
    p1 += responsabilities['1'][i] * observations[i][0]
p1 = p1 / np.sum(responsabilities['1'])
print(f'p1: {p1:.5f}')



print('\n')
mu2 = 0
for i in range(len(observations)):
    mu2 += responsabilities['2'][i] * observations[i][1:]
mu2 = mu2 / np.sum(responsabilities['2'])
print(f'mu2: {mu2}')
# print a string with mu2 in latex code:
latex_code = "mu2 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(mu2, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix}'
print(latex_code)


sigma2 = np.zeros((2, 2))
for i in range(len(observations)):
    array = np.expand_dims(observations[i][1:] - mu2, axis=1)
    multiplication = np.dot(array, array.T)
    sigma2 += responsabilities['2'][i] * multiplication
sigma2 = sigma2 / np.sum(responsabilities['2'])
# print sigma2 with 5 decimals
print(f'sigma2:\n ', np.round(sigma2, 5))
# print a string with sigma2 in latex code:
latex_code = "sigma2 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(sigma2, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix}'
print(latex_code)

p2 = 0
for i in range(len(observations)):
    p2 += responsabilities['2'][i] * observations[i][0]
p2 = p2 / np.sum(responsabilities['2'])
print(f'p2: {p2:.5f}')

pi1 = np.mean(responsabilities['1'])
print(f'\npi1: {pi1:.5f}')

pi2 = np.mean(responsabilities['2'])
print(f'pi2: {pi2:.5f}\n')


# Given a x_new observation, calculate the probability of belonging to each cluster
print('\n------------------------------X New -----------------------------\n')

x_new = np.array([1, 0.3, 0.7])

# Calculate the Gaussian and Bernoulli probabilities for each cluster
print(f'x_new: {x_new}, mu1: {mu1}, sigma1: {sigma1}, p1: {p1}, pi1: {pi1}')
gaussian_1 = multivariate_normal.pdf(x_new[1:], mean=mu1, cov=sigma1)
print(f'Gaussian 1 x_new: {gaussian_1:.5f}')
bernoulli_1 = bernoulli.pmf(x_new[0], p1)
print(f'Bernoulli 1 x_new: {bernoulli_1:.5f}')
responsabilities_1 = gaussian_1 * bernoulli_1 * pi1
print(f'Responsability 1 x_new: {responsabilities_1:.5f}\n')

gaussian_2 = multivariate_normal.pdf(x_new[1:], mean=mu2, cov=sigma2)
print(f'Gaussian 2 x_new: {gaussian_2:.5f}')
bernoulli_2 = bernoulli.pmf(x_new[0], p2)
print(f'Bernoulli 2 x_new: {bernoulli_2:.5f}')
responsabilities_2 = gaussian_2 * bernoulli_2 * pi2
print(f'Responsability 2 x_new: {responsabilities_2:.5f}\n')

# Normalize the responsibilities
sum_responsabilities = responsabilities_1 + responsabilities_2
responsabilities_1 /= sum_responsabilities
responsabilities_2 /= sum_responsabilities
print(f'Normalized responsability 1: {responsabilities_1:.5f}')
print(f'Normalized responsability 2: {responsabilities_2:.5f}')
if responsabilities_1 > responsabilities_2:
    print('The observation belongs to cluster 1\n')
else:
    print('The observation belongs to cluster 2\n')


# Do a hard assignment of the observations to the clusters
print('\n------------------------------ Silhouette-----------------------------\n')

clusters = []
for i in range(len(observations)):
    print('\n')
    print(f'Observation {i+1}: {observations[i]}')
    gaussian_1 = multivariate_normal.pdf(observations[i][1:], mean=mu1, cov=sigma1)
    print(f'Gaussian 1_{i+1}: {gaussian_1:.5f}')
    bernoulli_1 = bernoulli.pmf(observations[i][0], p1)
    print(f'Bernoulli 1_{i+1}: {bernoulli_1:.5f}')
    gaussian_2 = multivariate_normal.pdf(observations[i][1:], mean=mu2, cov=sigma2)
    print(f'Gaussian 2_{i+1}: {gaussian_2:.5f}')
    bernoulli_2 = bernoulli.pmf(observations[i][0], p2)
    print(f'Bernoulli 2_{i+1}: {bernoulli_2:.5f}')
    responsability_1 = gaussian_1 * bernoulli_1 
    print(f'Responsability 1_{i+1}: {responsability_1:.5f}')
    responsability_2 = gaussian_2 * bernoulli_2
    print(f'Responsability 2_{i+1}: {responsability_2:.5f}')
    # Normalize the responsibilities
    sum_responsabilities = responsability_1 + responsability_2
    responsability_1 /= sum_responsabilities
    responsability_2 /= sum_responsabilities
    print(f'Normalized responsability 1_{i+1}: {responsability_1:.5f}')
    print(f'Normalized responsability 2_{i+1}: {responsability_2:.5f}')
    if responsability_1 > responsability_2:
        clusters.append(1)
    else:
        clusters.append(2)
print(f'\nClusters: {clusters}\n\n')

# Sompute silhouette coefficient for both clusters assuming manhattan distance
# a(i) = average distance between i and all other points in the same cluster
# b(i) = average distance between i and all other points in the closest cluster
# s(i) = (b(i) - a(i)) / max(a(i), b(i))
# s = average(s(i))
# Print matrix with all distances
print('\nDistances matrix:\n')
for i in range(len(observations)):
    distances = []
    for j in range(len(observations)):
        distances.append(round(np.linalg.norm(observations[i] - observations[j], ord=1), 5))
    print(distances, '\n')

# Calculate a(i) for each point
a = []
for i in range(len(observations)):
    distances = []
    for j in range(len(observations)):
        if clusters[i] == clusters[j] and i != j:
            distances.append(np.linalg.norm(observations[i] - observations[j], ord=1))
    a.append(round(np.mean(distances), 5))
print(f'a: {a}')

# Calculate b(i) for each point
b = []
for i in range(len(observations)):
    distances = []
    for j in range(len(observations)):
        if clusters[i] != clusters[j]:
            distances.append(np.linalg.norm(observations[i] - observations[j], ord=1))
    b.append(round(np.mean(distances), 5))

print(f'b: {b}')


# Calculate s(i) for each point
s = []
for i in range(len(observations)):
    s.append(round((b[i] - a[i]) / max(a[i], b[i]), 5))
print(f's: {s}')

# Calculate s for cluster 1
s_1 = []
for i in range(len(observations)):
    if clusters[i] == 1:
        s_1.append(s[i])

print(f's_1: {np.mean(s_1):.5f}')

# Calculate s for cluster 2
s_2 = []
for i in range(len(observations)):
    if clusters[i] == 2:
        s_2.append(s[i])

print(f's_2: {np.mean(s_2):.5f}')

# Calculate s for all clusters
print(f's for solution: {round((np.mean(s_1) + np.mean(s_2) )/ 2, 5)}')

# Predict number of classes using purity of 0.75 
# Purity = (1/N) * sum(max(class_i))
# N = number of observations
# class_i = number of observations in class i
# max(class_i) = number of observations in class i with the most common class label








