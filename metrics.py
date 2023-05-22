import torch
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score
import sys

import warnings

warnings.filterwarnings('once')

def one_hot_encod(l):
    return [[0, 1] if j==1 else [1, 0] for j in l]

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = [] #pred
        self.t = [] #true

    def update(self, y, t): #pred, true
        self.y+=y
        self.t+=t
        
    def get_metrics(self, reduction='none'):
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])        
        tn, fp, fn, tp = self._process(self.y, self.t)
        rec = self.recall_new(self.y, self.t)
        pre = self.precision_new(self.y, self.t)
        acc = self.accuracy_new(self.y, self.t)
        if rec+pre!=0:
            f1 = 2*(rec*pre)/(rec+pre)
        else:
            f1 = 0
        return acc, pre, rec, f1, tn, fp, fn, tp

    def get_preds_labels(self):
        return self.y, self.t
    
    def _process(self, y, t):
        tn, fp, fn, tp = confusion_matrix(t, y).ravel()
        return tn, fp, fn, tp

    def recall_new(self, y, t):
        return recall_score(t, y)
    
    def precision_new(self, y, t):
        return precision_score(t, y)
    
    def accuracy_new(self, y, t):
        return accuracy_score(t, y)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
if __name__ == '__main__':
    test()