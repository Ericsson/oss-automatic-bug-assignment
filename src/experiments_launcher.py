# -*- coding: utf-8 -*-
"""
.. module:: experiments_launcher
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to launch 
              all the experiments related to a particular software 
              project (Mozilla Firefox, Eclipse JDT, etc.) 
              sequentially.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import abc
import logging

class ExperimentsLauncher(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self, \
                 raw_data_set_file_path, \
                 pre_processed_data_set_file_path, \
                 developers_dict_file_path=None, \
                 developers_list_file_path=None):
        """Constructor"""   
        # Path of the file containing the raw data set
        self.raw_data_set_file_path = raw_data_set_file_path
        
        # Path of the file containing one of the pre-processed data 
        # sets
        self.pre_processed_data_set_file_path = \
        pre_processed_data_set_file_path
        
        # Below, the path of the file which contains a dictionary 
        # related to the mappings of the developers
        self.developers_dict_file_path = developers_dict_file_path
        
        # Below, the path of the file which contains a list of the 
        # relevant distinct developers
        self.developers_list_file_path = developers_list_file_path
        
        # Will contain an instance of the class related to the first 
        # experiment
        self.experiment_1 = None

        # Will contain an instance of the class related to the second 
        # experiment        
        self.experiment_2 = None
        
        # Will contain an instance of the class related to the third 
        # experiment
        self.experiment_3 = None
        
        # Will contain an instance of the class related to the forth 
        # experiment
        self.experiment_4 = None
        

    @abc.abstractmethod
    def conduct_experiment_1(self):
        """This method runs the experiment 1"""
        self.experiment_1.conduct_experiment()
    
    @abc.abstractmethod
    def conduct_experiment_2(self):
        """This method runs the experiment 2"""
        self.experiment_2.conduct_experiment()
    
    @abc.abstractmethod    
    def conduct_experiment_3(self):
        """This method runs the experiment 3"""
        self.experiment_3.conduct_experiment()
        
    @abc.abstractmethod
    def conduct_experiment_4(self):
        """This method runs the experiment 4"""
        self.experiment_4.conduct_experiment()

    def conduct_experiments(self):
        """This method runs all the experiments
        
        Only the experiments related to the current instance will be
        launched.
        """
        self.conduct_experiment_1()
        self.remove_handlers_root_logger_object()
        self.conduct_experiment_2()
        self.remove_handlers_root_logger_object()
        self.conduct_experiment_3()
        self.remove_handlers_root_logger_object()
        self.conduct_experiment_4()
    
    @staticmethod
    def remove_handlers_root_logger_object():
        """Removes all handlers related to the root logger object"""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)