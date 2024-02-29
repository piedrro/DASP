import numpy as np


def dice_coefficient(y_true, y_pred):
    """
    Function to calculate Dice Coefficient.
    
    Arguments:
    y_true -- ground truth labels
    y_pred -- predicted labels
    
    Returns:
    dice -- Dice Coefficient score
    """
    # The '1' here prevents division by zero
    intersection = np.logical_and(y_true, y_pred).sum()
    dice = (2. * intersection + 1) / (np.sum(y_true) + np.sum(y_pred) + 1)
    
    return dice

def iou(y_true, y_pred):
    """
    Function to calculate Intersection Over Union.
    
    Arguments:
    y_true -- ground truth labels
    y_pred -- predicted labels
    
    Returns:
    iou -- Intersection Over Union score
    """
    # The '1' here prevents division by zero
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = (intersection + 1) / (union + 1)
    
    return iou