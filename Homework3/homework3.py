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
    X_new[i,1]=math.exp(-(aux1[0]**2 + aux1[1]**2)/2)
    aux2=np.array([[X[i][1]] - c_2[0], [X[i][2]] - c_2[1]])
    X_new[i,2]=math.exp(-(aux2[0]**2 + aux2[1]**2)/2)
    aux3=np.array([[X[i][1]] - c_3[0], [X[i][2]] - c_3[1]])
    X_new[i,3]=math.exp(-(aux3[0]**2 + aux3[1]**2)/2)

# print with 5 decimal places
print(np.around(X_new, decimals=5))

# we want to apply ridge regression with lambda=0.1 to the transformed dataset

W = np.linalg.inv(X_new.T.dot(X_new) + 0.1*np.identity(4)).dot(X_new.T).dot(Z)

# print("inverse = ", np.around(np.linalg.inv(X_new.T.dot(X_new) + 0.1*np.identity(4)), decimals=5))



# print with 5 decimal places
print(np.around(W, decimals=5))

# estimate the output value for the 4 observations in the dataset

Z_hat = X_new.dot(W)
print(np.around(Z_hat, decimals=5))

# compute the RMSE
RMSE = np.sqrt(np.sum((Z-Z_hat)**2)/len(Z))
print("RMSE = ", np.around(RMSE, decimals=5))
print("RMSE**2*4 = ", np.around(RMSE**2*4, decimals=5))