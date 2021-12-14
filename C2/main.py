from utils import plot_data, generate_data
import numpy as np
import matplotlib.pyplot as plt
"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    # X.shape = (n, m)
    n = X.shape[0]  # get the rows' size
    m = X.shape[1]  # get the columns' size
    # Initialize the weights and bias
    w = np.zeros(m)
    b = 0
    alpha = 0.01    # learning rate
    # Gradient Descent Algorithm
    # For Logistic Regression, we have f(x) = σ(x·w + b)
    for i in range(100000):
        #z = sigmoid_function(X.dot(w) + b)    # z.shape = (1, m)
        z = predict_logistic_regression(X, w, b)
        Xt = X.transpose()
        dz = z - t
        dw = Xt.dot(dz) / n
        db = np.sum(dz) / n
        w = w - alpha * dw
        b = b - alpha * db
    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    #  y =  σ(x·w + b)
    y = sigmoid_function(X.dot(w) + b)
    # treat the target 0/1 labels as real numbers
    # and classify a sample as positive if the predicted value is greater than or equal to 0.5
    t = np.zeros(X.shape[0])
    t[y >= 0.5] = 1
    return t

def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    X_aug = np.concatenate([X, np.ones([len(t), 1])], axis=1)
    w = leastSquares(X_aug, t)
    b = w[-1]
    w = w[0: -1]
    return w, b

def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    X_aug = np.concatenate([X, np.ones([X.shape[0], 1])], axis=1)
    w = np.append(w, b)
    y = X_aug.dot(w)
    # treat the target 0/1 labels as real numbers
    # and classify a sample as positive if the predicted value is greater than or equal to 0.5
    t = np.zeros(X.shape[0])
    t[y >= 0.5] = 1
    return t

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    # Accuracy = the number of correctly predicted / the number of total samples
    correct = sum(t == t_hat)
    total = len(t_hat)
    acc = correct / total
    return acc

def sigmoid_function(x):
    # σ(z) = 1/[1 + e^(-z)] is known as the sigmoid or logistic function
    sigmoid_func = 1 / (1 + np.exp(-x))
    return sigmoid_func

def leastSquares(X, Y):
    # In this function, X is always the input, Y is always the output
    # X: M x (d+1), Y: M x 1, where d=1 here
    # return weights w

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


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_A_linear.png')
    plt.title('dataset_A_linear')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_A_logistic.png')
    plt.title('dataset_A_logistic')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_B_linear.png')
    plt.title('dataset_B_linear')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_B_logistic.png')
    plt.title('dataset_B_logistic')

main()
