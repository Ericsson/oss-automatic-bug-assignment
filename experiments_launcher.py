# -*- coding: utf-8 -*-

import abc

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

    def conduct_experiments(self):
        """This method runs all the experiments
        
        Only the experiments related to the current instance will be
        launched.
        """
        self.experiment_1.conduct_experiment()
        self.experiment_2.conduct_experiment()
        self.experiment_3.conduct_experiment()
        self.experiment_4.conduct_experiment()        