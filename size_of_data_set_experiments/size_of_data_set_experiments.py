# -*- coding: utf-8 -*-

from size_of_data_set_normal_experiment import SizeOfDataNormalExperiment
from size_of_data_set_incremental_experiment import SizeOfDataIncrementalExperiment
import logging

if __name__ == "__main__":
    logging.basicConfig(filename="size_of_data_set_experiments.log", \
    filemode="w", level=logging.DEBUG)
    data_set_file = "../pre_processing_experiments/output_without_cleaning_without_stemming_without_lemmatizing_without_stop_words_removal_without_punctuation_removal_without_numbers_removal.json" # The path of the file which 
    # contains the pre-processed output
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers
    mhos_dict_file = "../../developers_dict.json" 
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    mhos_list_file = "../../developers_list.json" 
    size_of_data_normal_experiment = SizeOfDataNormalExperiment(data_set_file, \
    developers_dict_file, developers_list_file)
    
    size_of_data_incremental_experiment = SizeOfDataIncrementalExperiment(data_set_file, \
    developers_dict_file, developers_list_file)
    
    K = [4, 6, 8, 10, 15, 25, 50] # Number of folds
    
    # K = [4, 6] # Number of folds
    
    for k in K:
        size_of_data_normal_experiment.plot_or_save_learning_curve(k)
        size_of_data_incremental_experiment.plot_or_save_learning_curve(k)
