"""
Classes and functions that complement those of pytorch.

Currently: F beta loss functions.

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"
 

import torch
from torch import nn
import torch.nn.functional as F


def fbeta_loss(input, target, beta=1, average='macro'):
    """Multiclass F beta loss function when predicted values are probabilities.

    If arg average is 'macro' or 'weighted', retrurns the respective Fbeta
    score (single number). If average is None, returns (n_classes) array 
    of scores for each class. 

    Note: Also implemented in ml.fbeta_score_smooth(), but here uses torch 

    Arguments:
      - input: (n_samples x n_classes) predicted class probabilities
      - target: (n_samples) expected classes 
      - beta: beta factor
      - average: Fbeta mode, can be macro, weighted
    """

    # make confusion matrix
    cm = torch.vstack(
        [input[target==cl_ind].sum(axis=0)
         for cl_ind in range(input.shape[-1])])

    # get f scores for all classes separately
    score_class = (
        (1 + beta**2) * cm.diagonal()
        / (beta**2 * cm.sum(axis=1) + cm.sum(axis=0)))

    # average f scores
    if average is None:
        score = score_class
    elif average == 'macro':
        score = score_class.mean()
    elif average == 'weighted':
        score = (score_class * cm.sum(axis=1)).sum() / cm.sum()
    else:
        raise ValueError(
            f"Sorry, average value {average} is not implemented. "
            + "Currently implemented are 'macro' and 'weighted'.")

    return 1 - score

def f1_loss(input, target, average='macro'):
    """F1 loss function for multiclass
    """

    return fbeta_loss(input, target, beta=1, average=average)


class FbetaLoss():
    """F beta loss for multiclass.

    """

    def __init__(self, beta=1, average='macro'):
        self.beta = beta
        self.average = average

    def __call__(self, input, target):
        return fbeta_loss(input, target, beta=self.beta, average=self.average)
        
    def forward(self, input, target):
        return fbeta_loss(input, target, beta=self.beta, average=self.average)

class F1Loss():
    """F1 loss for multiclass.
    """
    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, input, target):
        return f1_loss(input, target, average=self.average)
        
    def forward(self, input, target):
        return f1_loss(input, target, average=self.average)
  
