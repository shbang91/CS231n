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

    num_data = X.shape[0]
    num_class = W.shape[1]
    
    for i in range(num_data):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        soft_max = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(soft_max[y[i]])
        
        for j in range(num_class):
            dW[:,j] += soft_max[j] * X[i]
        dW[:,y[i]] -= X[i]
        
    loss = 1 / num_data * loss + reg * np.sum(W * W)
    
    dW /= num_data
    dW += 2 * reg * W

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

    num_data = X.shape[0]
    num_class = W.shape[1]
    
    scores = X.dot(W)
    max_val = scores[[a for a in range(num_data)],np.argmax(scores,axis=1)]
    scores -= np.expand_dims(max_val,axis=1)
    row_sum = np.sum(np.exp(scores),axis=1)
    soft_max = np.exp(scores) / np.expand_dims(row_sum,axis=1)
    loss_element = soft_max[[a for a in range(num_data)],y]
    loss = 1 / num_data * np.sum(-np.log(loss_element)) + reg * np.sum(W * W)
    
    mod = soft_max 
    mod[range(num_data),y] -= 1.
    dW = X.T.dot(mod)
    dW /= num_data
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
