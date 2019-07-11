from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i].T
                dW[:,y[i]] -= X[i].T
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #added some codes above

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    scores = X.dot(W)
    correct_scores = scores[[x for x in range(X.shape[0])],list(y)]
    
    margins = scores - np.expand_dims(correct_scores,axis=1) + 1 
    loss = np.maximum(0,margins)
    loss[[x for x in range(X.shape[0])],list(y)] = 0.
    loss = np.sum(loss) / num_train + reg * np.sum(W * W)
    
    
      ##  loss = 0.5 / num_train * (np.sum(scores - np.expand_dims(correct_scores,axis=1) + np.ones((num_train,num_class))) + np.sum(np.absolute(scores - np.expand_dims(correct_scores,axis=1) + np.ones((num_train,num_class))))) + reg * np.sum(W*W)
    
      ##  print(X.dot(W)[[x for x in range(X.shape[0])],y].shape)
      ##  loss = 0.5 / X.shape[0] * (np.sum(X.dot(W) - np.expand_dims(y,axis=1) + np.ones((X.shape[0],W.shape[1]))) + np.sum(np.absolute(X.dot(W) - np.expand_dims(y,axis=1) + np.ones((X.shape[0],W.shape[1]))))) + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    margincount = np.zeros(margins.shape)
    margincount[margins>0] = 1
    margincount[np.arange(num_train),y] = 0
    margincount[np.arange(num_train),y] = -np.sum(margincount,axis=1)
    
    dW = X.T.dot(margincount)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
