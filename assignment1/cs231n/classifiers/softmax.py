from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = np.zeros((num_train, num_class))

    for i in range(num_train):
        scores[i, :] = X[i, :].dot(W)
        max_score = np.max(scores[i, :])
        loss += -scores[i, y[i]] + max_score + np.log(
            np.sum(np.exp(scores[i, :] - max_score)))

        for j in range(num_class):
            dW[:, j] += np.exp(scores[i, j] - max_score) / np.sum(
                np.exp(scores[i, :] - max_score)) * X[i, :]
        dW[:, y[i]] -= X[i, :]

    loss = loss / num_train + reg * np.sum(W * W)
    dW /= num_train + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    scores = X.dot(W)
    score_max = np.max(scores, axis=1)
    scores = scores - np.reshape(
        score_max, (num_train, -1))  #process to avoid instability
    soft_max = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1),
                                           (scores.shape[0], -1))
    soft_max_loss = np.exp(scores[np.arange(num_train), y]) / np.sum(
        np.exp(scores), axis=1)
    loss = np.sum(-np.log(soft_max_loss))
    loss = loss / num_train + reg * np.sum(W * W)

    soft_max[np.arange(soft_max.shape[0]), y] += -1  #true class gradient
    dW = X.T.dot(soft_max) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
