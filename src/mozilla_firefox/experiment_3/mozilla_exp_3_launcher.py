# -*- coding: utf-8 -*-
"""
.. module:: mozilla_classify_k_folds_time_series_feature_selection
   :platform: Unix, Windows
   :synopsis: This module contains a class used to conduct the third
              experiment of the thesis on the bug reports of Mozilla
              Firefox. The experiment consists mainly of comparing
              several feature selection techniques and selecting the
              best one.

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
from experiment_3.exp_3_launcher \
import Exp3Launcher

class MozillaExp3Launcher(Exp3Launcher):
    def __init__(self, data_set_file, developers_dict_file=None, \
                 developers_list_file=None):
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

    mozilla_exp_3_launcher = MozillaExp3Launcher( \
    data_set_file=data_set_file, \
    developers_dict_file=developers_dict_file, \
    developers_list_file=developers_list_file)
    mozilla_exp_3_launcher.conduct_experiment()

if __name__ == "__main__":
    main()