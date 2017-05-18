# -*- coding: utf-8 -*-
"""
.. module:: eclipse_size_of_data_set_experiments
   :platform: Unix, Windows
   :synopsis: This module contains a class used to conduct both sub 
              experiments of the preliminary experiment of the thesis
              on the bug reports of Eclipse JDT. The experiment 
              consists mainly of trying to find the optimal number 
              of bug reports that should be used to train a 
              classifier.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from eclipse_size_of_data_set_normal_experiment \
import EclipseSizeOfDataNormalExperiment
from eclipse_size_of_data_set_incremental_experiment \
import EclipseSizeOfDataIncrementalExperiment
from size_of_data_set_experiments.size_of_data_set_experiments \
import SizeOfDataExperiments

class EclipseSizeOfDataExperiments(SizeOfDataExperiments):
    
    def __init__(self, data_set_file, developers_dict_file=None, \
                 developers_list_file=None):
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__()
        self.size_of_data_normal_experiment = EclipseSizeOfDataNormalExperiment( \
        data_set_file, developers_dict_file, developers_list_file)
        self.size_of_data_incremental_experiment = EclipseSizeOfDataIncrementalExperiment( \
        data_set_file, developers_dict_file, developers_list_file)

def main():
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
    
    size_of_data_experiments = EclipseSizeOfDataExperiments( \
    data_set_file, developers_dict_file, developers_list_file)
    size_of_data_experiments.conduct_experiment()
    
if __name__ == "__main__":
    main()