# -*- coding: utf-8 -*-
"""
.. module:: mozilla_data_pre_processer
   :platform: Unix, Windows
   :synopsis: This module contains a class used to apply different
              combinations of pre-processing techniques on the sorted
              bug reports of Mozilla Firefox (the data should have
              been scrapped via the Scrapy library, in a JSON file and
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
from experiment_1.data_pre_processor \
import DataPreProcessor

# Pre-process the id, title, description and assigned-to fields of the
# aforementioned bug reports.

class MozillaDataPreProcessor(DataPreProcessor):

    case_sensitive_elements_to_remove = [
#         r"\bPM\b", # NO 2
#         r"n't\b", # NO 3
        r"'ll\b", # OK 4
#         r"'s\b", # NO 5
#         r"/\*", # NO 6
        r"\*/", # OK 7
        r"->", # OK 8
#         r"\|+", # NO 9
        r"\{+", # OK 10
        r"\}+", # OK 11
#         r"\[+", # NO 12
        r"\]+", # OK 13
#         r"\(+", # NO 14
#         r"\)+", # NO 15
        r";+", # OK 16
        r">+", # OK 17
        r"<+", # OK 18
        r"=+", # OK 19
#         r"'+", # NO 20
#         r"`+", # NO 21
        r"\"+", # OK 22
#         r"\++", # NO 23
#         r"\*+", # NO 24
        r"#+", # OK 25
#         r":+", # NO 26
#         r"\.{2,}", # NO 27
        r"\.+", # OK 28
#         r"/{2,}", # NO 29
        r"/+", # OK 30
#         r"-+", # NO 31
        r"\b\w\b" # OK 32
    ]

    case_insensitive_elements_to_remove = [
#         r"\bok\b", # NO 33
        r"\bsth\b" # OK 34
#         r"\bnotes?\b" # NO 35
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
    data_file = "../scrap_mozilla_firefox/sorted_brs.json"

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

    mozilla_data_pre_processor = MozillaDataPreProcessor(data_file, \
    clean_brs=clean_brs, use_stemmer=use_stemmer, \
    use_lemmatizer=use_lemmatizer, \
    stop_words_removal=stop_words_removal, \
    punctuation_removal=punctuation_removal, \
    numbers_removal=numbers_removal) # Instantiation of the
    # Pre-processer
    mozilla_data_pre_processor.clean_data() # We clean the data
    mozilla_data_pre_processor.write_data(output_data_file) # We write the
    # data into the given output file

    print("--- {} seconds ---".format(time.time() - start_time))

if __name__ == "__main__":
    main()