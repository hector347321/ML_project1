import numpy as np


def compute_mse(y, tx, w):
  
    e = y - tx @ w
    return float(0.5 * np.mean(e**2))


def compute_mse_gradient(y, tx, w):

    e = y - tx @ w
    return -tx.T @ e / y.shape[0]


def sigmoid(t):

    return np.exp(-np.logaddexp(0, -t))


def compute_logistic_loss(y, tx, w):

    t = tx @ w
    return float(np.mean(np.logaddexp(0, t) - y * t))


def compute_logistic_gradient(y, tx, w):

    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):

    w = np.array(initial_w, dtype=float) 
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
  
    w = np.array(initial_w, dtype=float)
    N = y.shape[0]
    for _ in range(max_iters):
        n = np.random.randint(N)
        grad = -(y[n] - tx[n] @ w) * tx[n]
        w = w - gamma * grad
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
  
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    w = np.linalg.solve(A, tx.T @ y)
    return w, compute_mse(y, tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
   
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
