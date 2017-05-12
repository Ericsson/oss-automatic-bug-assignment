# -*- coding: utf-8 -*-

from size_of_data_set_experiment import SizeOfDataExperiment
import numpy as np
import abc
        
class SizeOfDataIncrementalExperiment(SizeOfDataExperiment):
    
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