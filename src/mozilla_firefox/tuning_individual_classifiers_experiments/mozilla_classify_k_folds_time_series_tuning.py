# -*- coding: utf-8 -*-
"""
.. module:: mozilla_classify_k_folds_time_series_tuning
   :platform: Unix, Windows
   :synopsis: This module contains a class used to conduct the last 
              experiment of the thesis on the bug reports of Mozilla 
              Firefox. The experiment consists mainly of tuning 
              several classifiers and selecting the best performing 
              one.

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
from tuning_individual_classifiers_experiments.classify_k_folds_time_series_tuning \
import TuningIndividualClassifierGenericExperiment

class MozillaTuningIndividualClassifierGenericExperiment(TuningIndividualClassifierGenericExperiment):
    def __init__(self, data_set_file, developers_dict_file, \
                 developers_list_file):
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(data_set_file, True, False, \
                         developers_dict_file, developers_list_file)

def main():
    data_set_file = "../pre_processing_experiments/output_with_" + \
    "cleaning_without_stemming_with_lemmatizing_without_stop_" + \
    "words_removal_without_punctuation_removal_with_numbers_" + \
    "removal.json" # The path of the file which contains the 
    # pre-processed output
        
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers
    developers_dict_file = None
    
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    developers_list_file = None
    
    tuning_individual_classifier_generic_experiment = \
    MozillaTuningIndividualClassifierGenericExperiment( \
    data_set_file=data_set_file, \
    developers_dict_file=developers_dict_file, \
    developers_list_file=developers_list_file)
    tuning_individual_classifier_generic_experiment \
    .conduct_experiment()    

if __name__ == "__main__":
    main()