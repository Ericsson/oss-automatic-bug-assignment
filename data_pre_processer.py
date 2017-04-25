# -*- coding: utf-8 -*-

import json
from html.parser import HTMLParser
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
import string
import csv
import sys
import time
import os
import inspect

"""Pre-process BRs

Pre-process the id, heading, observation and assignee fields of the 
BRs.
"""

class DataPreProcesser:

    case_sensitive_elements_to_remove = [
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"\b\b",
        r"Â",
        r"~+",
        r"\|+",
        r"/+",
        r"\{+",
        r"\}+",
        r"\[+",
        r"\]+",
        r"\(+",
        r"\)+",
        r";+",
        r"->",
        r">+",
        r"<+",
        r"=+",
        r"'+",
        r"`+",
        r"\"+",
        r"\++",
        r"#+",
        r"\d+:\d+",
        r":{2,}",
        r"\b\d+(\.\d+)+\.?\b",
        r"\.{2,}",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        r"\d{2}:\d{2}:\d{2}(\.\d{3})?",
        r"\d{4}-\d{2}-\d{2}",
        r"-+"
    ]
    
    case_insensitive_elements_to_remove = [
        r"\b\b",
        r"\b",
        r"\b\b",
        r"\b\b",
        r"\b"
    ]
    
    def __init__(self, data_file, clean_brs=True, \
    use_stemmer=True, use_lemmatizer= False, \
    stop_words_removal=True, punctuation_removal=True, \
    numbers_removal=True):
        """Constructor"""
        self.content = None # Will contain the content of the file 
        # with all BRs
        self.html_parser = None # Will contain an instance of a HTML 
        # parser
        self.stemmer = None # Will contain the stemmer which will be 
        # used
        self.lemmatizer = None # Will contain the lemmatizer which 
        # will be used
        self.en_stop_words = None # Will contain an english stop word
        # list
        self.output = [] # Will contain the pre-processed data to dump 
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        # Will contain the name of the file which contains the BRs
        self.data_file = os.path.join(self._current_dir, data_file) 
        if clean_brs:
            # Instantiation of a HTMLParser
            self.html_parser = HTMLParser() 
        if use_stemmer:
            # Instantiation of a Porter Stemmer
            self.stemmer = PorterStemmer()
        elif use_lemmatizer:
            # Instantiation of a lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            pass
        if stop_words_removal:
            # Get an English stop word list
            self.en_stop_words = stopwords.words('english') 
        self.punctuation_removal = punctuation_removal # To know 
        # whether or not punctuation should be removed
        self.numbers_removal = numbers_removal # To know whether or 
        # not numbers should be removed
        if numbers_removal:
            self.case_sensitive_elements_to_remove.append(r"\b\d+\b")

    def clean_data(self):
        """Method to clean the data

        It cleans the id, heading, observation and assignee fields.
        """
        print("BR number:") # Debug
        i = 0  
        # Below, we open the data file
        with open(self.data_file, encoding="utf8") as f:
            self.content = csv.DictReader(f, delimiter=";")
            for row in self.content:
                # print(row["W"]) # Debug
                i += 1
                print(i)
                # Then, we clean them and add them to a list
                self.output.append({
                    "br_id": self \
                    .__clean_br_id(row["X"]),
                    "observation": self \
                    .__clean_observation( \
                        row["Y"] \
                        ),
                    "heading": self \
                    .__clean_heading(row["Z"]),
                    "assignee": self \
                    .__clean_assignee( \
                        row["A"] \
                        )
                })
                # if i == 1000:
                #     break

    def write_data(self, data_file):
        """Method to write the data

        It writes the data into a JSON file which path is given in 
        parameter.
        """
        with open(data_file, 'w') as output_file:
            json.dump(self.output, output_file)
            
    def __clean_string(self, string_to_clean):
        """Method to clean a given string"""
        if string_to_clean is None:
            string_to_clean = ""
        else:
            if self.html_parser is not None:
                # We should clean the string
                string_to_clean = self \
                .__escape_html_char_and_remove_space(string_to_clean)
                string_to_clean = self.__remove_url(string_to_clean)
                for el_to_remove in self \
                .case_sensitive_elements_to_remove:
                    string_to_clean = re.sub(el_to_remove, " ", \
                    string_to_clean)
                for el_to_remove in self \
                .case_insensitive_elements_to_remove:
                    string_to_clean = re.sub(el_to_remove, " ", \
                    string_to_clean, flags=re.I)
                string_to_clean = self \
                .__replace_newline_char_and_strip(string_to_clean)
                re.sub(r"\s+", " ", string_to_clean)
        tokens = word_tokenize(string_to_clean)
        if self.stemmer is not None:
            # We should stem the string to clean
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatizer is not None:
            tokens_pos = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(token, \
            pos=self.get_wordnet_pos(pos)) for token, pos \
            in tokens_pos]
        if self.en_stop_words:
            # We should remove the stop words
            tokens = [token for token in tokens if token.lower() \
            not in self.en_stop_words]
        if self.punctuation_removal:
            # We should remove the punctuation characters        
            tokens = [token for token in tokens \
            if not all(char in string.punctuation for char in token)]
        return tokens

    def __clean_heading(self, heading):
        """Method to clean a given heading"""
        return self.__clean_string(heading)

    def __clean_observation(self, observation):
        """Method to clean a given observation"""
        # Below, we clean, tokenize and stem the observation field
        return self.__clean_string(observation)

    def __clean_br_id(self, br_id):
        """Method to clean a given BR-id"""
        return self.__escape_html_char_and_remove_space(br_id)

    def __escape_html_char_and_remove_space(self, data):
        """Method to escape HTML chars and remove spaces"""
        data = "" if data is None else data
        if self.html_parser is not None:
            data = self.html_parser.unescape(data)
            return data.replace("\u00a0", "")
        else:
            return data

    def __clean_assignee(self, assignee):
        """Method to clean an assignee"""
        return self.__escape_html_char_and_remove_space(assignee)

    def __replace_newline_char_and_strip(self, data):
        """Method to replace a '\n' char and to strip a string"""
        return self.__replace_newline_char_by_space(data).strip()

    def __replace_newline_char_by_space(self, data):
        """Method to replace a '\n'"""
        return data.replace("\n", " ")

    def __remove_url(self, data):
        """Method to remove any URL"""
        data = re \
        .sub(r'(https|http|file)?:\/\/\/?(\/|\w|\.|\?|\=|\&|\%|\-)*\b' \
        , '', data)		
        return data
        
    def get_wordnet_pos(self, treebank_pos):
        if treebank_pos.startswith('J'):
            return wordnet.ADJ
        elif treebank_pos.startswith('V'):
            return wordnet.VERB
        elif treebank_pos.startswith('N'):
            return wordnet.NOUN
        elif treebank_pos.startswith('R') or \
        treebank_pos.startswith('WR'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

if __name__ == "__main__":
    """Main program"""
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # Decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True

    start_time = time.time() # We get the time expressed in seconds 
    # since the epoch

    clean_brs = True
    use_stemmer = False
    use_lemmatizer = True
    stop_words_removal = True
    punctuation_removal = True
    numbers_removal = True

    data_file = "X.csv" # Path of the file containing 
    # the data
    
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
    
    data_pre_processer = DataPreProcesser(data_file, \
    clean_brs=clean_brs, use_stemmer=use_stemmer, \
    use_lemmatizer=use_lemmatizer, \
    stop_words_removal=stop_words_removal, \
    punctuation_removal=punctuation_removal, \
    numbers_removal=numbers_removal) # Instantiation of the 
    # Pre-processer
    data_pre_processer.clean_data() # We clean the data
    data_pre_processer.write_data(output_data_file) # We write the 
    # data into the given output file
    
    print("--- {} seconds ---".format(time.time() - start_time)