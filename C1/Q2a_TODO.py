#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y = None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here
    y_new = X.dot(w)
    # Since we normalize features and output by the equation
    # y_hat = (y_new - mean(y_new)) / std(y_new)
    y_hat = (y_new - np.mean(y_new)) / np.std(y_new)

    # the training loss is J = (1/2M)·sum(y_norm - t_norm)^2
    loss  = 1/2 * (1/len(y)) * np.sum(np.abs(y - y_hat) ** 2)

    # The measure of success (the lower, the better) is E = (1/M)·sum(|y - t|)^2
    y_old = (y * std_y) + mean_y
    risk  = (1 / len(y_old)) * np.sum(np.abs(y_old - y_new))
    
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]

    # initialization
    w = np.ones([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val   = []

    w_best    = None
    risk_best = 10000
    epoch_best= 0
    
    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train/batch_size)) ):
            
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            # M should be the number of samples in a batch
            # theta = theta - (1/M) * alpha * x * (h(x) - y)
            X_batch_T = X_batch.transpose()
            w -= (1 / len(y_batch)) * alpha * X_batch_T.dot((y_hat_batch - y_batch))

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        ave_loss_this_epoch = loss_this_epoch / int(np.ceil(N_train/batch_size))
        losses_train.append(ave_loss_this_epoch)
        # 2. Perform validation on the validation test by the risk
        _, _, risk_val = predict(X_val, w, y_val)
        risks_val.append(risk_val)
        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val <= risk_best:
          # Since MaxIter = 100, the range of MaxIter is 0 to 99
          # we need to count epoch start from 1
          epoch_best = epoch + 1
          risk_best = risk_val
          w_best = w

    # Return some variables as needed
    # the training loss by averaging loss_this_epoch
    # validation on the validation test by the risk
    # the best validation epoch, risk, and the weights
    return losses_train, risks_val, epoch_best, risk_best, w_best



############################
# Main code starts here
############################

# Load data. This is the only allowed API call from sklearn
X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)



# Augment feature
X_ = np.concatenate( ( np.ones([X.shape[0],1]), X ), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y  = np.std(y)

y = (y - np.mean(y)) / np.std(y)

#print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val   = X_[300:400]
y_val   = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha   = 0.001      # learning rate
batch_size   = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay




# TODO: Your code here
losses_train, risks_val, epoch_best, risk_best, w_best = train(X_train, y_train, X_val, y_val)

# Perform test by the weights yielding the best validation performance
_, loss_test, risk_test = predict(X_test, w_best, y = y_test)

# Report numbers and draw plots as required. 
# report three numbers 
print("1. The number of epoch that yields the best validation performance: ", epoch_best)
print("2. The validation performance (risk) in that epoch: ", risk_best)
print("3. The test performance (risk) in that epoch: ", risk_test)

# report two plots
# where x-axis is the number of epochs, and y-axis is training loss and validation risk, respectively
# 1. The learning curve of the training loss
plt.figure()
plt.plot(losses_train)
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.title('The learning curve of the training loss')
plt.tight_layout()
plt.savefig('2(a)_The learning curve of the training loss' + '.jpg')

# 2. The learning curve of the validation risk
plt.figure()
plt.plot(risks_val)
plt.xlabel('epoch')
plt.ylabel('validation risk')
plt.title('The learning curve of the validation risk')
plt.tight_layout()
plt.savefig('2(a)_The learning curve of the validation risk' + '.jpg')

plt.show()
