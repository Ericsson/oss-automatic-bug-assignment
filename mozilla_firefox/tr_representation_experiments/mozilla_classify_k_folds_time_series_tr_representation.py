# -*- coding: utf-8 -*-

import os
import inspect
import logging

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from tr_representation_experiments.classify_k_folds_time_series_tr_representation \
import TRRepresentationExperiment

class MozillaTRRepresentationExperiment(TRRepresentationExperiment):
    def __init__(self, data_set_file, developers_dict_file=None, \
        developers_list_file=None):
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(data_set_file, developers_dict_file, \
                         developers_list_file)

def main():
    data_set_file = "../pre_processing_experiments/output_with_" + \
    "cleaning_without_stemming_without_lemmatizing_with_stop_" + \
    "words_removal_with_punctuation_removal_with_numbers_" + \
    "removal.json" # The path of the file which contains the 
    # pre-processed output
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers
    developers_dict_file = None
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    developers_list_file = None
    
    tr_representation_experiment = MozillaTRRepresentationExperiment( \
    data_set_file=data_set_file, \
    developers_dict_file=developers_dict_file, \
    developers_list_file=developers_list_file)
    tr_representation_experiment.conduct_experiment()

if __name__ == "__main__":
    main()