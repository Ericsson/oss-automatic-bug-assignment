# -*- coding: utf-8 -*-
"""
.. module:: eclipse_data_pre_processer
   :platform: Unix, Windows
   :synopsis: This module contains a class used to apply different 
              combinations of pre-processing techniques on the sorted 
              bug reports of Eclipse JDT (the data should have been 
              scrapped via the Scrapy library, in a JSON file and 
              should have been sorted before).

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

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

# Pre-process the id, title, description and assigned-to fields of the 
# aforementioned bug reports.

class EclipseDataPreProcesser(DataPreProcesser):
    
    case_sensitive_elements_to_remove = [
        r"\bPM\b", # OK 2
        r"n't\b", # OK 3
#         r"'ll\b", # NO 4
#         r"'s\b", # NO 5
#         r"/\*", # NO 6
#         r"\*/", # NO 7
        r"\.java\b", # OK 8
        r"\bx+\b", # OK 9
        r"->", # OK 10
#         r"\|+", # NO 11
        r"\{+", # OK 12
        r"\}+", # OK 13
        r"\[+", # OK 14
        r"\]+", # OK 15
#         r"\(+", # NO 16
#         r"\)+", # NO 17
        r";+", # OK 18
        r">+", # OK 19
        r"<+", # OK 20
#         r"=+", # NO 21
        r"'+", # OK 22
#         r"`+", # NO 23
        r"\"+", # OK 24
        r"\++", # OK 25
        r"\*+", # OK 26
        r"#+", # OK 27
#         r":+", # NO 28
        r"\.{2,}", # OK 29
        r"\.+", # OK 30
        r"/{2,}", # OK 31
        r"/+", # OK 32
#         r"-+", # NO 33
        r"\b\w\b" # OK 34
#         r"\b[A-Z0-9]{7}\b", # NO 35
#         r"\b[A-Z]{2,4}\b" # NO 36
    ]
    
    case_insensitive_elements_to_remove = [
#         r"\bok\b", # NO 37
#         r"\bsth\b", # NO 38
#         r"\bnotes?\b" # NO 39
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
    clean_brs = False
    use_stemmer = False
    use_lemmatizer = False
    stop_words_removal = False
    punctuation_removal = False
    numbers_removal = False
    
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