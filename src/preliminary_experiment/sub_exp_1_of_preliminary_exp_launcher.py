# -*- coding: utf-8 -*-
"""
.. module:: size_of_data_set_incremental_experiment
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to conduct 
              the first sub experiment of the preliminary experiment 
              of the thesis. The experiment consists mainly of trying 
              to find the optimal number of bug reports that should 
              be used to train a classifier. In the context of the 
              first sub experiment, all the folds, except the oldest 
              one, are used to evaluate the performance of the 
              classifier (cf. Master's Thesis). 

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import numpy as np
import abc
import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from sub_exp_of_preliminary_exp_launcher \
import SubExpOfPreliminaryExpLauncher
        
class SubExp1OfPreliminaryExpLauncher(SubExpOfPreliminaryExpLauncher):
    
    @abc.abstractmethod
    def __init__(self, data_set_file, developers_dict_file, \
                 developers_list_file):
        super().__init__(data_set_file, developers_dict_file, \
                         developers_list_file)
        self._type = "incremental"
    
    def _yield_indices_for_learning_curve(self, K=33):
        super()._yield_indices_for_learning_curve(K)
        number_of_instances = self._X.shape[0]
        indices = super()._custom_linspace(0, number_of_instances, K+1)
        for i in range(len(indices)-2, 0, -1):
            for j in range(i):
                yield np.asarray(range(indices[j], indices[i])), \
                np.asarray(range(indices[i], indices[i+1]))
                
    def _generate_list_indices_for_learning_curve(self, K=33):
        super()._generate_list_indices_for_learning_curve(K)
        number_of_instances = self._X.shape[0]
        indices = super()._custom_linspace(0, number_of_instances, K+1)
        train_indices = []
        test_indices = []
        for i in range(len(indices)-2, 0, -1):
            for j in range(i):
                train_indices.append(list(range(indices[j], indices[i])))
                test_indices.append(list(range(indices[i], indices[i+1])))
        return train_indices, test_indices 
        
    def plot_or_save_learning_curve(self, K=33, save_file=True):
        super().plot_or_save_learning_curve(K, save_file)