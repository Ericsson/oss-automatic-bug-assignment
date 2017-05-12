# -*- coding: utf-8 -*-

import abc
import logging

class SizeOfDataExperiments(abc.ABC):
    
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
            self.size_of_data_normal_experiment.plot_or_save_learning_curve(k)
            self.size_of_data_incremental_experiment.plot_or_save_learning_curve(k)