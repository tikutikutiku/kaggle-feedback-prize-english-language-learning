import numpy as np

def calc_metric(pred, gt):
    '''
    pred : (num_data, num_labels)
    gt : (num_data, num_labels)
    '''
    score = np.sqrt(np.mean((pred - gt)**2, axis=0))
    score = score.mean()
    return score