
import numpy as np

def sigmoid(y):
    return 1.0/(1.0 + np.exp(-y))

def calculate_gradient(theta, X, Y):
    n = Y.size
    return X.T@(sigmoid(X@theta) - Y)/n

def gradient_descent(theta, X, Y, alpha = 0.1, iterations = 100, tol = 1e-7):
    for i in range(iterations):
        grad = calculate_gradient(theta, X, Y)
        theta -= alpha*grad

        if np.linalg.norm(grad) < tol:
            break
    return theta


def predict_probs(theta, X):
    return sigmoid(X@theta)

def predict(X, theta, threshold = 0.5 ):
    return (predict_probs(theta, X) >= threshold).ashtype(int)

