# -*- coding: utf-8 -*-
"""
.. module:: size_of_data_set_normal_experiment
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to conduct 
              the second sub experiment of the preliminary experiment 
              of the thesis. The experiment consists mainly of trying 
              to find the optimal number of bug reports that should 
              be used to train a classifier. In the context of the 
              second sub experiment, only the latest fold is used to 
              evaluate the performance of the classifier (cf. Master's 
              Thesis). 

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import numpy as np
import abc
import os
import inspect
        
current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from size_of_data_set_experiment import SizeOfDataExperiment
        
class SizeOfDataNormalExperiment(SizeOfDataExperiment):
    
    @abc.abstractmethod
    def __init__(self, data_set_file, developers_dict_file, developers_list_file):
        super().__init__(data_set_file, developers_dict_file, developers_list_file)
        self._type = "normal"
        
    def _yield_indices_for_learning_curve(self, K=33):
        super()._yield_indices_for_learning_curve(K)
        number_of_instances = self._X.shape[0]
        indices = super()._custom_linspace(0, number_of_instances, K+1)
        beginning_test_set_idx = len(indices)-2 
        for i in range(beginning_test_set_idx):
            yield np.asarray(range(indices[i], \
            indices[beginning_test_set_idx])), \
            np.asarray(range(indices[beginning_test_set_idx], \
            indices[beginning_test_set_idx+1]))
                
    def _generate_list_indices_for_learning_curve(self, K=33):
        super()._generate_list_indices_for_learning_curve(K)
        number_of_instances = self._X.shape[0]
        indices = super()._custom_linspace(0, number_of_instances, K+1)
        train_indices = []
        test_indices = []
        beginning_test_set_idx = len(indices)-2 
        for i in range(beginning_test_set_idx):
            train_indices.append(list(range(indices[i], \
            indices[beginning_test_set_idx])))
            test_indices.append(list(range(indices[beginning_test_set_idx], \
            indices[beginning_test_set_idx+1])))            
        return train_indices, test_indices 
        
    def plot_or_save_learning_curve(self, K=33, save_file=True):
        super().plot_or_save_learning_curve(K, save_file)