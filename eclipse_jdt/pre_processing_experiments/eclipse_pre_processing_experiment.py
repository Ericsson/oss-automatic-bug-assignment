# -*- coding: utf-8 -*-

import os
import inspect
import logging

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from eclipse_data_pre_processer import EclipseDataPreProcesser
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir) 
from pre_processing_experiments.pre_processing_experiment \
import PreProcessingExperiment

class EclipsePreProcessingExperiment(PreProcessingExperiment):
    
    def __init__(self, data_file, developers_dict_file, \
    developers_list_file, clean_brs=False, use_stemmer=False, \
    use_lemmatizer=False, stop_words_removal=False, \
    punctuation_removal=False, numbers_removal=False):
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(data_file, developers_dict_file, \
        developers_list_file, clean_brs=False, use_stemmer=False, \
        use_lemmatizer=False, stop_words_removal=False, \
        punctuation_removal=False, numbers_removal=False)
        
    def generate_output_file(self):
        """Generates a specific pre-processed data set file"""
        # Instantiation of a specific pre-processor
        self._data_pre_processer = EclipseDataPreProcesser( \
        self.data_file, clean_brs=self.clean_brs, \
        use_stemmer=self.use_stemmer, \
        use_lemmatizer=self.use_lemmatizer, \
        stop_words_removal=self.stop_words_removal, \
        punctuation_removal=self.punctuation_removal, \
        numbers_removal=self.numbers_removal)
        # Call of the super-class same method
        super().generate_output_file()
        
def main():
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers
    developers_dict_file = None
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    developers_list_file = None
    
    clean_brs = False
    use_stemmer = False
    use_lemmatizer = False
    stop_words_removal = False
    punctuation_removal = False
    numbers_removal = False
    
    data_file = "../scrap_eclipse_jdt/sorted_brs.json" # Path of the file containing the data
    
    pre_processing_experiment = EclipsePreProcessingExperiment( \
    data_file=data_file, \
    developers_dict_file=developers_dict_file, developers_list_file=developers_list_file, \
    clean_brs=clean_brs, \
    use_stemmer=use_stemmer, \
    use_lemmatizer=use_lemmatizer, \
    stop_words_removal=stop_words_removal, \
    punctuation_removal=punctuation_removal, \
    numbers_removal=numbers_removal)
    
    pre_processing_experiment.conduct_experiment()
        
if __name__ == "__main__":
    main()