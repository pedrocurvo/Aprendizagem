import numpy as np
import matplotlib.pyplot as plt
import math

# dataset

X=np.array([[1,  0.7, -0.3],
            [1,  0.4,  0.5],
            [1, -0.2,  0.8],
            [1, -0.4,  0.3]])
Z=np.array([[0.8],
            [0.6],
            [0.3],
            [0.3]])

# transforming the dataset

c_1=np.array([[0], [0]])
c_2=np.array([[1], [-1]])
c_3=np.array([[-1], [1]])

# We now get 3 new variables in each observation

X_new = np.empty([len(X), 4])

for i in range(len(X)):
    X_new[i,0]=1
    aux1=np.array([[X[i][1]] - c_1[0], [X[i][2]] - c_1[1]])
    X_new[i,1]=math.exp(-(aux1[0][0]**2 + aux1[1][0]**2)/2)
    aux2=np.array([[X[i][1]] - c_2[0], [X[i][2]] - c_2[1]])
    X_new[i,2]=math.exp(-(aux2[0][0]**2 + aux2[1][0]**2)/2)
    aux3=np.array([[X[i][1]] - c_3[0], [X[i][2]] - c_3[1]])
    X_new[i,3]=math.exp(-(aux3[0][0]**2 + aux3[1][0]**2)/2)

# print with 5 decimal places
print('Transformed dataset = \n',
    np.around(X_new, decimals=5))

# we want to apply ridge regression with lambda=0.1 to the transformed dataset

W = np.linalg.inv(X_new.T.dot(X_new) + 0.1*np.identity(4)).dot(X_new.T).dot(Z)

# print("inverse = ", np.around(np.linalg.inv(X_new.T.dot(X_new) + 0.1*np.identity(4)), decimals=5))



# print with 5 decimal places
print( 'W = \n',
    np.around(W, decimals=5))

# estimate the output value for the 4 observations in the dataset

Z_hat = X_new.dot(W)
print('Z_hat = \n',
    np.around(Z_hat, decimals=5))

# compute the RMSE
RMSE = np.sqrt(np.sum((Z-Z_hat)**2)/len(Z))
print("RMSE = ", np.around(RMSE, decimals=5))
print("RMSE**2*4 = ", np.around(RMSE**2*4, decimals=5), '\n \n \n')



# new exercise
# gradient descent update

w_11=np.array([[1,1,1,1],
               [1,1,2,1],
               [1,1,1,1]])
w_21=np.array([[1,4,1],
               [1,1,1]])
w_31=np.array([[1,1],
               [3,1],
               [1,1]])
b_11=np.array([[1],
            [1],
            [1]])
b_21=np.array([[1],
            [1]])
b_31=np.array([[1],
            [1],
            [1]])

x_10=np.array([[1], [1], [1], [1]])
t_1=np.array([[0], [1], [0]])
x_20=np.array([[1], [0], [0], [-1]])
t_2=np.array([[1], [0], [0]])

# Porpagation of x_11

print(np.around((w_11.dot(x_10) + b_11) * 0.5 - 2, decimals=5))
x_11 = np.tanh((w_11.dot(x_10) + b_11) * 0.5 - 2)
print("x_11 = \n", np.around(x_11, decimals=5))
z_11 = (w_11.dot(x_10) + b_11)
print("z_11 = \n", np.around(z_11, decimals=5))


print(np.around((w_21.dot(x_11) + b_21) * 0.5 - 2, decimals=5))
x_12 = np.tanh((w_21.dot(x_11) + b_21) * 0.5 - 2)
print("x_12 = \n", np.around(x_12, decimals=5))
z_12 = (w_21.dot(x_11) + b_21)
print("z_12 = \n", np.around(z_12, decimals=5))

print(np.around((w_31.dot(x_12) + b_31) * 0.5 - 2, decimals=5))
x_13 = np.tanh((w_31.dot(x_12) + b_31) * 0.5 - 2)
print("x_13 = \n", np.around(x_13, decimals=5))
z_13 = (w_31.dot(x_12) + b_31)
print("z_13 = \n", np.around(z_13, decimals=5))

# Propagation of x_21

print(np.around((w_11.dot(x_20) + b_11) * 0.5 - 2, decimals=5))
x_21 = np.tanh((w_11.dot(x_20) + b_11) * 0.5 - 2)
print("x_21 = \n", np.around(x_21, decimals=5))
z_21 = (w_11.dot(x_20) + b_11)
print("z_21 = \n", np.around(z_21, decimals=5))

print(np.around((w_21.dot(x_21) + b_21) * 0.5 - 2, decimals=5))
x_22 = np.tanh((w_21.dot(x_21) + b_21) * 0.5 - 2)
print("x_22 = \n", np.around(x_22, decimals=5))
z_22 = (w_21.dot(x_21) + b_21)
print("z_22 = \n", np.around(z_22, decimals=5))

print(np.around((w_31.dot(x_22) + b_31) * 0.5 - 2, decimals=5))
x_23 = np.tanh((w_31.dot(x_22) + b_31) * 0.5 - 2)
print("x_23 = \n", np.around(x_23, decimals=5))
z_23 = (w_31.dot(x_22) + b_31)
print("z_23 = \n", np.around(z_23, decimals=5))

# updated weights for w_3

delta_13 = -0.1 * 0.5 * np.multiply((x_13-t_1),(1 - np.tanh(z_13 * 0.5 - 2) * np.tanh(z_13 * 0.5 - 2))).dot(x_12.T)
# print a string with delta_13 in latex code:
latex_code = "delta_13 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(delta_13, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

delta_23 = -0.1 * 0.5 * np.multiply((x_23-t_2),(1 - np.tanh(z_23 * 0.5 - 2) * np.tanh(z_23 * 0.5 - 2))).dot(x_22.T)
# print a string with delta_23 in latex code:
latex_code = "delta_23 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(delta_23, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

deltaw_3 = delta_13 + delta_23
# print a string with deltaw_3 in latex code:
latex_code = "deltaw_3 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(deltaw_3, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

# new w_3
w_32 = w_31 + deltaw_3
# print a string with w_32 in latex code:
latex_code = "w_32 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(w_32, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)


# updated weights for w_2


delta_12 = -0.1 * 0.5 * 0.5 * w_31.T.dot((x_13 - t_1) * (1 - np.tanh(z_13 * 0.5 - 2) * np.tanh(z_13 * 0.5 - 2))) * (1 - np.tanh(z_12 * 0.5 - 2) * np.tanh(z_12 * 0.5 - 2)).dot(x_11.T)
# print a string with delta_12 in latex code:
latex_code = "delta_12 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(delta_12, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

delta_22 = -0.1 * 0.5 * 0.5 * w_31.T.dot((x_23 - t_2) * (1 - np.tanh(z_23 * 0.5 - 2) * np.tanh(z_23 * 0.5 - 2))) * (1 - np.tanh(z_22 * 0.5 - 2) * np.tanh(z_22 * 0.5 - 2)).dot(x_21.T)
# print a string with delta_22 in latex code:
latex_code = "delta_22 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(delta_22, decimals=10), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

deltaw_2 = delta_12 + delta_22
# print a string with deltaw_2 in latex code:
latex_code = "deltaw_2 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(deltaw_2, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

# new w_2
w_22 = w_21 + deltaw_2
# print a string with w_22 in latex code:
latex_code = "w_22 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(w_22, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)


# updated weights for w_1


delta_11 = -0.1 * 0.5 * 0.5 * 0.5 * w_21.T.dot( w_31.T.dot((x_13 - t_1) * (1 - np.tanh(z_13 * 0.5 - 2) * np.tanh(z_13 * 0.5 - 2))) * (1 - np.tanh(z_12 * 0.5 - 2) * np.tanh(z_12 * 0.5 - 2))) * (1 - np.tanh(z_11 * 0.5 - 2) * np.tanh(z_11 * 0.5 - 2)).dot(x_10.T)
# print a string with delta_11 in latex code:
latex_code = "delta_11 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(delta_11, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)

delta_21 = -0.1 * 0.5 * 0.5 * 0.5 * w_21.T.dot( w_31.T.dot((x_23 - t_2) * (1 - np.tanh(z_23 * 0.5 - 2) * np.tanh(z_23 * 0.5 - 2))) * (1 - np.tanh(z_22 * 0.5 - 2) * np.tanh(z_22 * 0.5 - 2))) * (1 - np.tanh(z_21 * 0.5 - 2) * np.tanh(z_21 * 0.5 - 2)).dot(x_20.T)
# print a string with delta_21 in latex code:
latex_code = "delta_21 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(delta_21, decimals=10), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix} '
print(latex_code)


deltaw_1 = delta_11 + delta_21
# print a string with deltaw_1 in latex code:
latex_code = "deltaw_1 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(deltaw_1, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '  \end{bmatrix}'
print(latex_code)

# new w_1
w_12 = w_11 + deltaw_1
# print a string with w_12 in latex code:
latex_code = "w_12 in lateX code = " + '\\begin{bmatrix} ' + np.array2string(np.around(w_12, decimals=5), separator=' & ').replace('\n', r' \\ ' ).replace('[','').replace(']', '').replace('& \\', '\\') + '\end{bmatrix}'
print(latex_code)