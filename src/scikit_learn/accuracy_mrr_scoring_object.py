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
    """It is a scorer callable function.
    
    This function is used to compute the values of both the accuracy 
    and MRR metrics.
    
    :param estimator: The scikit-learn estimator to use.
    :type estimator: estimator object.
    :param X: Training vectors, where n_samples is the number of 
    samples and n_features is the number of features.
    :type X: {array-like, sparse matrix}, shape = [n_samples, 
    n_features].
    :param y: Target values.
    :type y: array-like.
    :returns: array -- the array contains the values of both the 
    accuracy and MRR metrics. If the value of the MRR metric can not 
    be computed, None or -1 will be added to the array.
    """
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