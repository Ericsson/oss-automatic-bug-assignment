# -*- coding: utf-8 -*-
"""
.. module:: accuracy_mrr_scoring_object
   :platform: Unix, Windows
   :synopsis: This module contains a scorer callable function used to 
              compute the values of both the accuracy and MRR metrics.
              The function is mainly used in the forth experiment of 
              the thesis.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from utilities import print_log

def accuracy_mrr_scoring_object(estimator, X, y):
    accuracy = np.mean(estimator.predict(X) == y)
    mrr = None
    found_function = False                      
    try:
        if callable(getattr(estimator, "predict_proba")):
#                                 print_log(self._model.classes_)
#                                 print_log(self._model.predict_proba(X_val))
            lb = LabelBinarizer()        
            _ = lb.fit_transform(estimator.classes_)
#                                 print_log(lb.classes_)
#                                 print_log(y_classes_bin)
#                                 print_log(lb.transform(["x"]))  
            y_bin = lb.transform(y)
            mrr = label_ranking_average_precision_score( \
            y_bin, \
            estimator.predict_proba(X))
            found_function = True
    except AttributeError:
        pass
    except ValueError as e:
        found_function = True
        print_log("The MRR score will be set to -1: {}".format(e))
        mrr = -1
    try:
        if not found_function and callable(getattr(estimator, "decision_function")):
#                                 print_log(self._model.classes_)
#                                 print_log(self._model.decision_function(X_val))
            lb = LabelBinarizer()        
            _ = lb.fit_transform(estimator.classes_)
#                                 print_log(lb.classes_)
#                                 print_log(y_classes_bin)
#                                 print_log(lb.transform(["x"]))  
            y_bin = lb.transform(y)
            mrr = label_ranking_average_precision_score( \
            y_bin, \
            estimator.decision_function(X))
            found_function = True
    except AttributeError:
        pass
    except ValueError as e:
        found_function = True
        print_log("The MRR score will be set to -1: {}".format(e))
        mrr = -1
    return np.array([accuracy, mrr])