import numpy as np
from random import shuffle

# key points: 1. softmax_loss_vectorizednp.random.choice()
#             2. np.random.choice()


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
  number , dim = X.shape
  classes = W.shape[1]
  f = np.dot(X, W)  # N,C
  f_sub = np.max(f, axis=1, keepdims=True)
  softmax = np.exp(f-f_sub)/np.sum(np.exp(f-f_sub), axis=1, keepdims=True)  # N,C
  y_one_hot = np.zeros_like(softmax)
  y_one_hot[np.arange(number), y] = 1
  dw_every = np.zeros_like(W)
  for i in range(number):
    for j in range(classes):
      dw_every[:, j] = (softmax[i, j]-y_one_hot[i, j])*X[i, :]
      loss += -(y_one_hot[i, j]*np.log(softmax[i, j]))
    dW += dw_every
  # 正则化
  loss = loss/number
  loss = loss + 0.5*reg*np.sum(W*W)
  dW = dW/number
  dW = dW + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)  # D,C

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  number, dim = X.shape
  classes = W.shape[1]
  f = np.dot(X, W)  # N,C
  f_sub = np.max(f, axis=1, keepdims=True)
  softmax = np.exp(f - f_sub) / np.sum(np.exp(f - f_sub), axis=1, keepdims=True)  # N,C
  y_one_hot = np.zeros_like(softmax)
  y_one_hot[np.arange(number), y] = 1

  loss += -np.sum(y_one_hot*np.log(softmax))
  dW += np.dot(X.T, softmax-y_one_hot)
  # 正则化
  loss = loss / number
  loss = loss + 0.5 * reg * np.sum(W * W)
  dW = dW / number
  dW = dW + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW
