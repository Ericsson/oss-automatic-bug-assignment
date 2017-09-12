# -*- coding: utf-8 -*-
"""
.. module:: mozilla_experiments_launcher
   :platform: Unix, Windows
   :synopsis: This module contains a class used to launch all the 
              experiments related to Mozilla Firefox sequentially.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from experiment_1.mozilla_exp_1_launcher import MozillaExp1Launcher
from tr_representation_experiments \
.mozilla_classify_k_folds_time_series_tr_representation import \
MozillaTRRepresentationExperiment
from experiment_3.mozilla_exp_3_launcher import MozillaExp3Launcher
from tuning_individual_classifiers_experiments \
.mozilla_classify_k_folds_time_series_tuning import \
MozillaTuningIndividualClassifierGenericExperiment
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, current_dir)
from experiments_launcher import ExperimentsLauncher

class MozillaExperimentsLauncher(ExperimentsLauncher):
    
    def __init__(self, \
                 raw_data_set_file_path, \
                 pre_processed_data_set_file_path, \
                 developers_dict_file_path=None, \
                 developers_list_file_path=None):
        """Constructor"""
        # Call of the constructor of the super-class
        super().__init__(raw_data_set_file_path, \
                         pre_processed_data_set_file_path, \
                         developers_dict_file_path, \
                         developers_list_file_path)
            
    def conduct_experiment_1(self):
        """This method runs the experiment 1"""
        # Instantiation of the class related to the first experiment
        self.experiment_1 = MozillaExp1Launcher( \
        data_file=self.raw_data_set_file_path, \
        developers_dict_file=self.developers_dict_file_path, \
        developers_list_file=self.developers_list_file_path
        )
        super().conduct_experiment_1()
    
    def conduct_experiment_2(self):
        """This method runs the experiment 2"""
        # Instantiation of the class related to the second experiment
        self.experiment_2 = MozillaTRRepresentationExperiment( \
        data_set_file=self.pre_processed_data_set_file_path, \
        developers_dict_file=self.developers_dict_file_path, \
        developers_list_file=self.developers_list_file_path)
        super().conduct_experiment_2()
    
    def conduct_experiment_3(self):
        """This method runs the experiment 3"""
        # Instantiation of the class related to the third experiment
        self.experiment_3 = MozillaExp3Launcher( \
        data_set_file=self.pre_processed_data_set_file_path, \
        developers_dict_file=self.developers_dict_file_path, \
        developers_list_file=self.developers_list_file_path)
        super().conduct_experiment_3()
        
    def conduct_experiment_4(self):
        """This method runs the experiment 4"""
        # Instantiation of the class related to the forth experiment
        self.experiment_4 = \
        MozillaTuningIndividualClassifierGenericExperiment( \
        data_set_file=self.pre_processed_data_set_file_path, \
        developers_dict_file=self.developers_dict_file_path, \
        developers_list_file=self.developers_list_file_path)
        super().conduct_experiment_4()
        
def main():
    """The main function of this module
    
    This function is used to launch all the experiments related to the
    Mozilla Firefox project.
    """
    # Path of the file containing the raw data set
    raw_data_set_file_path = \
    "../scrap_mozilla_firefox/sorted_brs.json" 
    
    # Path of the file containing one of the pre-processed data sets
    pre_processed_data_set_file_path = "../pre_processing_" + \
    "experiments/output_with_cleaning_without_stemming_without_" + \
    "lemmatizing_with_stop_words_removal_with_punctuation_" + \
    "removal_with_numbers_removal.json"
    
    # Instantiation of the launcher of this module
    mozilla_experiments_launcher = \
    MozillaExperimentsLauncher(raw_data_set_file_path, \
                               pre_processed_data_set_file_path)
    
    # We launch all the experiments related to the aforementioned 
    # launcher
    mozilla_experiments_launcher.conduct_experiments()
if __name__ == "__main__":
    main() # Call of the main function of this module