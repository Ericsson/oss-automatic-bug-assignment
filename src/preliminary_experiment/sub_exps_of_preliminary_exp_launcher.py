# -*- coding: utf-8 -*-
"""
.. module:: size_of_data_set_experiments
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to conduct 
              both sub experiments of the preliminary experiment of 
              the thesis. The experiment consists mainly of trying to 
              find the optimal number of bug reports that should be 
              used to train a classifier.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import abc
import logging
import os

class SubExpsOfPreliminaryExpLauncher(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self):       
        self._K = [4, 6, 8, 10, 15, 25, 50] # Number of folds    
        # self._K = [4, 6] # Number of folds
        
        log_file = os.path.join(self._current_dir, \
        "size_of_data_set_experiments.log")
        logging.basicConfig(filename=log_file, filemode="w", \
                            level=logging.DEBUG)
        
    def conduct_experiment(self):
        for k in self._K:
            self.sub_exp_1_of_preliminary_exp_launcher.plot_or_save_learning_curve(k)
            self.sub_exp_2_of_preliminary_exp_launcher.plot_or_save_learning_curve(k)