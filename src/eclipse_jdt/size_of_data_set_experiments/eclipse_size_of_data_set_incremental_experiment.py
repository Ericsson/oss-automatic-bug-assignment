# -*- coding: utf-8 -*-
"""
.. module:: size_of_data_set_incremental_experiment
   :platform: Unix, Windows
   :synopsis: This module contains a class used to conduct the first 
              sub experiment of the preliminary experiment of the 
              thesis on the bug reports of Eclipse JDT. The experiment
              consists mainly of trying to find the optimal number of 
              bug reports that should be used to train a classifier. 
              In the context of the first sub experiment, all the 
              folds, except the oldest one, are used to evaluate the 
              performance of the classifier (cf. Master's Thesis). 

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect
import logging

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from size_of_data_set_experiments.size_of_data_set_incremental_experiment \
import SizeOfDataIncrementalExperiment

class EclipseSizeOfDataIncrementalExperiment(SizeOfDataIncrementalExperiment):
    
    def __init__(self, data_set_file, developers_dict_file, developers_list_file):
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(data_set_file, developers_dict_file, developers_list_file)
        
def main():
    logging.basicConfig(filename="size_of_data_set_incremental_experiment.log", \
    filemode="w", level=logging.DEBUG)
    
    data_set_file = "../pre_processing_experiments/output_" + \
    "without_cleaning_without_stemming_without_lemmatizing_" + \
    "without_stop_words_removal_without_punctuation_removal_" + \
    "without_numbers_removal.json" # The path of the file which 
    # contains the pre-processed output
    
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers    
    developers_dict_file = None
    
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    developers_list_file = None
    
    size_of_data_incremental_experiment = EclipseSizeOfDataIncrementalExperiment( \
    data_set_file, developers_dict_file, developers_list_file)
    
    K = 4 # Number of folds
    size_of_data_incremental_experiment.plot_or_save_learning_curve(K)  
        
if __name__ == "__main__":
    main()