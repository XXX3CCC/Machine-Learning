# -*- coding: utf-8 -*-


import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit

def readMNISTdata():

    with open('t10k-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels




def predict(X, W, b, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    y = X.dot(W)
    # Such computation is known as (k-way) softmax: z = softmax(y)
    z = softmax_function(y)
    t_hat = np.argmax(z, axis=1)
    t_arange = np.arange(len(t))
    z_log = np.log(z[t_arange, t])
    loss = - np.mean(z_log)
    # Accuracy = the number of correctly predicted / the number of total samples
    t_hat_shaped = t_hat.reshape(-1, 1)
    accuracy = get_accuracy(t_hat_shaped, t)
    acc = accuracy[0]
    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val, X_test, t_test):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
   
    #TODO Your code here
    M_train = X_train.shape[1]
    M_val   = X_val.shape[1]
    # Initialize weights and bias
    W = np.random.random((M_train, 10))
    b = np.random.random(10)
    W_best = None
    b_best = None
    # setting
    epoch = 0
    validation_acc_best = -10000
    test_acc_best = -10000
    losses_train = []       # The training cross-entropy loss
    validation_p = []   # The validation performance (accuracy) 
    test_p = []         # The test performance (accuracy) 

    for epoch in range(MaxEpoch):

        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train/batch_size)) ):

            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            # stochastic gradient descent (SGD)
            # y = X·w + b
            y = X_batch.dot(W) + b
            z = softmax_function(y)
            yl = len(y_batch)
            y_hat_batch = np.zeros((yl, 10))
            for i in range(len(y)):
                y_hat_batch[i, y_batch[i]] = 1
            Xt = X_batch.transpose()
            dz = z - y_hat_batch
            M = 1 / X_batch.shape[0]
            dw = M * Xt.dot(dz)
            db = M * np.sum(dz)
            W = W - alpha * dw
            b = b - alpha * db

            z_log = np.log(z)
            loss_batch = - np.mean(y_hat_batch * z_log)
            loss_this_epoch += loss_batch
            
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        ave_loss_this_epoch = loss_this_epoch / int(np.ceil(N_train/batch_size))
        losses_train.append(ave_loss_this_epoch)
        # 2. Get the validation performance (accuracy) and the test performance (accuracy) by predict
        _, _, _, validation_acc = predict(X_val, W, b, t_val)
        _, _, _, test_acc = predict(X_test, W, b, t_test)
        validation_p.append(validation_acc)
        test_p.append(test_acc)
        # 3. Keep track of the best validation epoch, weight, bias, validation performance and test performance
        if validation_acc_best < validation_acc:
            epoch_best = epoch
            W_best = W
            b_best = b
            validation_acc_best = validation_acc
            test_acc_best = test_acc
            
    return epoch_best, W_best, b_best, validation_acc_best, test_acc_best, losses_train, validation_p

def softmax_function(z):
    # We define a scoring function z = W·x + b
    # We make the scoring function normalized y_i = exp(z_i) / sum[exp(z_j)]
    # Calculate the maximum value of each row
    row_max = np.max(z)
    # The maximum value should be subtracted from each row element to prevent overflow
    dz = z - row_max
    z_exp = np.exp(dz)
    for i in range(len(z)):
        zi = z_exp[i]
        total = np.sum(zi)
        zi /= total
    return z_exp

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    # Accuracy = the number of correctly predicted / the number of total samples
    correct = sum(t == t_hat)
    total = len(t_hat)
    acc = correct / total
    return acc



##############################
#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()
X_test = X_test / 256

print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)



N_class = 10

alpha   = 0.1      # learning rate
batch_size   = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


epoch_best, W_best, b_best, validation_acc_best, test_acc_best, losses_train, validation_p = train(X_train, t_train, X_val, t_val, X_test, t_test)

#_, _, _, acc_test = predict(X_test, W_best, t_test)


# Without changing the the default hyperparameters, we report three numbers
print('The number of epoch that yields the best validation performance is ', epoch_best)
print('The validation performance (accuracy) in that epoch is ', validation_acc_best)
print('The test performance (accuracy) in that epoch is ', test_acc_best)

# Without changing the the default hyperparameters, we report two plots
# The learning curve of the training cross-entropy loss,
plt.figure()
plt.plot(losses_train)
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.title('The learning curve of the training cross-entropy loss')
plt.tight_layout()
plt.savefig('The_learning_curve_of_the_training_cross-entropy_loss' + '.png')

# The learning curve of the validation accuracy.
plt.figure()
plt.plot(validation_p)
plt.xlabel('epoch')
plt.ylabel('validation accuracy')
plt.title('The learning curve of the validation accuracy')
plt.tight_layout()
plt.savefig('The_learning_curve_of_the_validation_accuracy' + '.png')



