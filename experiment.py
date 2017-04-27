# -*- coding: utf-8 -*-

from sklearn.model_selection import TimeSeriesSplit
from utilities import load_data_set, build_data_frame, print_log
import numpy as np
import abc
import json

class Experiment(abc.ABC):

    @abc.abstractmethod
    def __init__(self, developers_dict_file=None, \
                 developers_list_file=None):
        self._data_set_file = None # Used to store the path of the 
        # data set file each model will use to be trained

        # TO DO: Modify the line below later  
        self._developers_dict_file = developers_dict_file

        # TO DO: Modify the line below later
        self._developers_list_file = developers_list_file   
        
        self._tscv = TimeSeriesSplit(n_splits=10) # Used to store a 
        # reference to the object which will allow us to perform a 
        # customized version of cross validation
    
        self._train_set = None # Used to store a reference to the 
        # training set
        self._test_set = None # Used to store a reference to the 
        # test set
    
        self._current_dir = None
        
        self._df = None
        
        # Below, there is a dictionary used to save the cleaned 
        # results to a JSON file
        self._results_to_save_to_a_file = {}
    
    def _build_data_set(self):
        # First we load the data set
        json_data = load_data_set(self._data_set_file)
        
        developers_dict_data = None
#         TO DO
#         load_developers_mappings(self._developers_dict_file)
        developers_list_data = None
#         TO DO
#         load_distinct_developers_list(self._developers_list_file)
        
        # TO DO
        # Then, we build a data frame using the loaded data set, the 
        # loaded developers mappings, the loaded distinct developers.
        self._df = build_data_frame(json_data, developers_dict_data, \
                                    developers_list_data)
        
        print_log("Splitting the data set") # Debug
        # self._df = self._df[-30000:]
        self._train_set, self._test_set = np.split(self._df, \
        [int(.9*len(self._df))])
        print_log("Shape of the initial Data Frame") # Debug
        print_log(self._df.shape) # Debug
        print_log(self._df['class'].value_counts(normalize=True))
        print_log("Shape of the training set") # Debug
        print_log(self._train_set.shape) # Debug
        print_log(self._train_set['class'].value_counts(normalize=True))
        print_log("Shape of the test set") # Debug
        print_log(self._test_set.shape) # Debug
        print_log(self._test_set['class'].value_counts(normalize=True))
                
    def _write_df(self):
        # We dump the data frame
        self._df.to_csv("pre_processed_data.csv")
        
    @abc.abstractmethod
    def conduct_experiment(self):
        self._save_cleaned_results()
        
    def _save_cleaned_results(self):
        """Method to write the cleaned results

        It writes the cleaned results into a JSON file which path is 
        an attribute of the object.
        """
        with open(self._cleaned_results_file_name, 'w') as output_file:
            json.dump(self._results_to_save_to_a_file, output_file, \
                      indent=4)