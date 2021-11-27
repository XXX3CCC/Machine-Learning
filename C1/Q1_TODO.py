#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def rotate(data, degree):
    # data: M x 2
    theta = np.pi/180 * degree
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]] )# rotation matrix
    return np.dot(data, R.T)


def leastSquares(X, Y):
    # In this function, X is always the input, Y is always the output
    # X: M x (d+1), Y: M x 1, where d=1 here
    # return weights w
    
    # TODO: YOUR CODE HERE
    # closed form solution by matrix-vector representations only
    # The transpose of X is X^T
    Xt = X.transpose()
    # Multiply X^T and X:
    XXt = Xt.dot(X)
    # Compute the (multiplicative) inverse of a matrix
    XXt_inv = np.linalg.inv(XXt)
    # Multiply the inverse of XXt and X^T:
    XXt_inv_Xt = XXt_inv.dot(Xt)
    # The global optimum: w = [(X^T·X)^(-1)]·X^T·Y
    w = XXt_inv_Xt.dot(Y)

    return w


def model(X, w):
    # X: M x (d+1)
    # w: d+1
    # return y_hat: M x 1

    # TODO: YOUR CODE HERE
    y_hat = X.dot(w)

    return y_hat


def generate_data(M, var1, var2, degree):

    # data generate involves two steps:
    # Step I: generating 2-D data, where two axis are independent
    # M (scalar): The number of data samples
    # var1 (scalar): variance of a
    # var2 (scalar): variance of b
    
    mu = [0, 0]

    Cov = [[var1, 0],
           [0,  var2]]

    data = np.random.multivariate_normal(mu, Cov, M)
    # shape: M x 2

    plt.figure()
    plt.scatter(data[:,0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.tight_layout()
    plt.savefig('data_ab_'+str(var2)+'.jpg')


    # Step II: rotate data by 45 degree counter-clockwise,
    # so that the two dimensions are in fact correlated


    data = rotate(data, degree)
    plt.tight_layout()
    plt.figure()
    # plot the data points
    plt.scatter(data[:,0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')

    # plot the line where data are mostly generated around
    X_new = np.linspace(-5, 5, 100, endpoint=True).reshape([100,1])

    Y_new = np.tan(np.pi/180*degree)*X_new
    plt.plot(X_new, Y_new, color="blue", linestyle='dashed')
    plt.tight_layout()
    plt.savefig('data_xy_'+str(var2)+ '_' + str(degree) + '.jpg')
    return data

###########################
# Main code starts here
###########################
# Q1(1)
# Settings
# With settings M = 5000, var1 = 1, var2 = 0.3, degree = 45, report the weight and bias for x2y and y2x regressions.
M = 5000
var1 = 1
var2 = 0.3
degree = 45

data = generate_data(M, var1, var2, degree)

##########
# Training the linear regression model predicting y from x (x2y)
Input  = data[:,0].reshape((-1,1)) # M x d, where d=1
Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
Output = data[:,1].reshape((-1,1)) # M x 1


w_x2y = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1

print('The result of Q1(1):\n')

print('Predicting y from x (x2y): weight='+ str(w_x2y[0,0]), 'bias = ', str(w_x2y[1,0]))

# # Training the linear regression model predicting x from y (y2x)

Input  = data[:,1].reshape((-1,1)) # M x d, where d=1
Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
Output = data[:,0].reshape((-1,1)) # M x 1

w_y2x = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1
print('Predicting x from y (y2x): weight='+ str(w_y2x[0,0]), 'bias = ', str(w_y2x[1,0]))


# Q1(2)
def generate_data_another(M, var1, var2, degree):
    mu = [0, 0]
    Cov = [[var1, 0],
           [0,  var2]]
    data = np.random.multivariate_normal(mu, Cov, M) # shape: M x 2
    # rotate data by 45 degree counter-clockwise
    data = rotate(data, degree)
    return data

# settings remain intact: M = 5000, var1 = 1, degree = 45
# Three plots of regression models in a row, 
# each with var2 = 0.1, 0.3, 0.8, respectively
M = 5000
var1 = 1
var2 = [0.1, 0.3, 0.8]
degree = 45
# Return evenly spaced values within a given interval
# Since plt.xlim(-4, 4), plt.ylim(-4, 4)
x = np.arange(-4, 4, 0.01).reshape((-1,1))
fig, ax = plt.subplots(figsize=(20, 4))

for i in range(len(var2)):

    # plot the data
    # Add an Axes to the current figure or retrieve an existing Axes
    plt.subplot(1, 3, i+1)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('var2 = ' + str(var2[i]))

    data = generate_data_another(M, var1, var2[i], degree)

    # Training the linear regression model predicting y from x (x2y)
    Input  = data[:,0].reshape((-1,1)) # M x d, where d=1
    Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
    Output = data[:,1].reshape((-1,1)) # M x 1
    x_aug = np.concatenate([x, np.ones([len(x), 1])], axis=1)

    w_x2y = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1

    x2y = plt.plot(x, model(x_aug, w_x2y), color = "red")

    # Training the linear regression model predicting x from y (y2x)
    Input  = data[:,1].reshape((-1,1)) # M x d, where d=1
    Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
    Output = data[:,0].reshape((-1,1)) # M x 1

    w_y2x = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1

    y2x = plt.plot(x, model(x_aug, w_y2x), color = "green")

    plt.legend([x2y[0], y2x[0]], ["x2y", "y2x"])

    plt.tight_layout()
    plt.savefig('Regression_model_' + str(var2) + '_' + str(degree) + '.jpg')

#plt.show()


# Q1(4)
# set var2 = 0.1, but experiment with different rotation degrees
M = 5000
var1 = 1
var2 = 0.1
# design a controlled experimental protocol
degree = [15, 45, 75]
# Return evenly spaced values within a given interval
# Since plt.xlim(-4, 4), plt.ylim(-4, 4)
x = np.arange(-4, 4, 0.01).reshape((-1,1))
fig, ax = plt.subplots(figsize=(20, 4))

for i in range(len(degree)):

    # plot the data
    # Add an Axes to the current figure or retrieve an existing Axes
    plt.subplot(1, 3, i+1)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('degree = ' + str(degree[i]))

    data = generate_data_another(M, var1, var2, degree[i])

    # Training the linear regression model predicting y from x (x2y)
    Input  = data[:,0].reshape((-1,1)) # M x d, where d=1
    Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
    Output = data[:,1].reshape((-1,1)) # M x 1
    x_aug = np.concatenate([x, np.ones([len(x), 1])], axis=1)

    w_x2y = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1

    x2y = plt.plot(x, model(x_aug, w_x2y), color = "red")

    # Training the linear regression model predicting x from y (y2x)
    Input  = data[:,1].reshape((-1,1)) # M x d, where d=1
    Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1) # M x (d+1) augmented feature
    Output = data[:,0].reshape((-1,1)) # M x 1

    w_y2x = leastSquares(Input_aug, Output) # (d+1) x 1, where d=1

    y2x = plt.plot(x, model(x_aug, w_y2x), color = "green")

    plt.legend([x2y[0], y2x[0]], ["x2y", "y2x"])

    plt.tight_layout()
    plt.savefig('Regression_model_' + str(var2) + '_' + str(degree) + '.jpg')


plt.show()