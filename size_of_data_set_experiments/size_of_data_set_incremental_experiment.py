# -*- coding: utf-8 -*-

from size_of_data_set_experiment import SizeOfDataExperiment
import numpy as np
import logging
        
class SizeOfDataIncrementalExperiment(SizeOfDataExperiment):
    def __init__(self, data_set_file, developers_dict_file, developers_list_file):
        super().__init__(data_set_file, developers_dict_file, developers_list_file)
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
    
if __name__ == "__main__":
    logging.basicConfig(filename="size_of_data_set_incremental_experiment.log", \
    filemode="w", level=logging.DEBUG)
    data_set_file = "../pre_processing_experiments/output_without_cleaning_without_stemming_without_lemmatizing_without_stop_words_removal_without_punctuation_removal_without_numbers_removal.json" # The path of the file which 
    # contains the pre-processed output
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers
    developers_dict_file = "../../developers_dict.json" 
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    developers_list_file = "../../developers_list.json" 
    size_of_data_incremental_experiment = SizeOfDataIncrementalExperiment(data_set_file, \
    developers_dict_file, developers_list_file)
    
    K = 4 # Number of folds
    size_of_data_incremental_experiment.plot_or_save_learning_curve(K)
