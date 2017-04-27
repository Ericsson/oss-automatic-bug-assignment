# -*- coding: utf-8 -*-

import os
import inspect
import time

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from pre_processing_experiments.data_pre_processer \
import DataPreProcesser

"""Pre-process the bug reports of the Eclipse JDT project

Pre-process the id, title, description and assigned-to fields of the 
aforementioned bug reports.
"""

class EclipseDataPreProcesser(DataPreProcesser):
    
    case_sensitive_elements_to_remove = [
#         r"\b\b",
#         r"\b\b",
#         r"Â",
#         r"~+",
#         r"\|+",
#         r"/+",
#         r"\{+",
#         r"\}+",
#         r"\[+",
#         r"\]+",
#         r"\(+",
#         r"\)+",
#         r";+",
#         r"->",
#         r">+",
#         r"<+",
#         r"=+",
#         r"'+",
#         r"`+",
#         r"\"+",
#         r"\++",
#         r"#+",
#         r"\d+:\d+",
#         r":{2,}",
#         r"\b\d+(\.\d+)+\.?\b",
#         r"\.{2,}",
#         r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
#         r"\d{2}:\d{2}:\d{2}(\.\d{3})?",
#         r"\d{4}-\d{2}-\d{2}",
#         r"-+"
    ]
    
    case_insensitive_elements_to_remove = [
#         r"\b\b",
#         r"\b"
    ]
    
    def __init__(self, data_file, clean_brs=True, use_stemmer=True, \
                 use_lemmatizer= False, stop_words_removal=True, \
                 punctuation_removal=True, numbers_removal=True):
        # Call of the constructor of the super-class
        super().__init__(data_file, clean_brs, use_stemmer, \
                         use_lemmatizer, stop_words_removal, \
                         punctuation_removal, numbers_removal)
        
def main():
    """Main program"""
    start_time = time.time() # We get the time expressed in seconds 
    # since the epoch

    current_dir = os.path.dirname(os.path.abspath( \
    inspect.getfile(inspect.currentframe())))

    # Path of the file containing the data set
    data_file = "../scrap_eclipse_jdt/sorted_brs.json" 
    
    # We build an absolute path
    data_file = os.path.join(current_dir, data_file)

    # The flags which are used to initialize the pre-processor
    clean_brs = True
    use_stemmer = True
    use_lemmatizer = False
    stop_words_removal = True
    punctuation_removal = True
    numbers_removal = True
    
    # Below, we are giving a relevant name to the output file
    clean_brs_string = "" if clean_brs else "out" 
    use_stemmer_string = "" if use_stemmer else "out"
    use_lemmatizer_string = "" if use_lemmatizer else "out"
    stop_words_removal_string = "" if stop_words_removal else "out"
    punctuation_removal_string = "" if punctuation_removal else "out"
    numbers_removal_string = "" if numbers_removal else "out"   
    output_data_file = "output_with{}_cleaning_".format(clean_brs_string) + \
    "with{}_stemming_".format(use_stemmer_string) + \
    "with{}_lemmatizing_".format(use_lemmatizer_string) + \
    "with{}_stop_words_removal_".format(stop_words_removal_string) + \
    "with{}_punctuation_removal_".format(punctuation_removal_string) + \
    "with{}_numbers_removal.json".format(numbers_removal_string)
    
    # We build an absolute path
    output_data_file = os.path.join(current_dir, output_data_file)
    
    data_pre_processer = EclipseDataPreProcesser(data_file, \
    clean_brs=clean_brs, use_stemmer=use_stemmer, \
    use_lemmatizer=use_lemmatizer, \
    stop_words_removal=stop_words_removal, \
    punctuation_removal=punctuation_removal, \
    numbers_removal=numbers_removal) # Instantiation of the 
    # Pre-processer
    data_pre_processer.clean_data() # We clean the data
    data_pre_processer.write_data(output_data_file) # We write the 
    # data into the given output file
    
    print("--- {} seconds ---".format(time.time() - start_time))

if __name__ == "__main__":
    main()