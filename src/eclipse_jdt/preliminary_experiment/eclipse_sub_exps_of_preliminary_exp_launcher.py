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
from eclipse_sub_exp_1_of_preliminary_exp_launcher \
import EclipseSubExp1OfPreliminaryExpLauncher
from eclipse_sub_exp_2_of_preliminary_exp_launcher \
import EclipseSubExp2OfPreliminaryExpLauncher
from preliminary_experiment.sub_exps_of_preliminary_exp_launcher \
import SubExpsOfPreliminaryExpLauncher

class EclipseSubExpsOfPreliminaryExpLauncher(SubExpsOfPreliminaryExpLauncher):
    
    def __init__(self, data_set_file, developers_dict_file=None, \
                 developers_list_file=None):
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__()
        self.sub_exp_1_of_preliminary_exp_launcher = \
        EclipseSubExp1OfPreliminaryExpLauncher(data_set_file, \
                                               developers_dict_file, \
                                               developers_list_file)
        self.sub_exp_2_of_preliminary_exp_launcher = \
        EclipseSubExp2OfPreliminaryExpLauncher(data_set_file, \
                                               developers_dict_file, \
                                               developers_list_file)

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
    
    eclipse_sub_exps_of_preliminary_exp_launcher = \
    EclipseSubExpsOfPreliminaryExpLauncher(data_set_file, \
                                           developers_dict_file, \
                                           developers_list_file)
    eclipse_sub_exps_of_preliminary_exp_launcher.conduct_experiment()
    
if __name__ == "__main__":
    main()