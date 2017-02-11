# -*- coding: utf-8 -*

import json
from HTMLParser import HTMLParser
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
import string

"""Pre-process some scrapped Eclipse JDT bug reports

Clean the fields of the scrapped Eclipse JDT bug reports.
"""

class DataPreProcesser:
    """Manage the pre-processing of the bug reports in a file"""

    json_data = None # Will contain the loaded JSON bug reports data
    html_parser = None # Will contain an instance of a HTML parser
    data_file = None # Will contain the name of the file which
    # contains the scrapped data
    stemmer = None # Will contain the stemmer which will be used
    en_stop_words = None # Will contain an English stop word list

    def __init__(self, data_file):
        """Constructor"""
        # Below, we open the data file
        with open(data_file) as json_data:
            self.json_data = json.load(json_data) # We load its
            # content
            self.html_parser = HTMLParser() # Instantiation of a
            # HTMLParser
            self.stemmer = PorterStemmer() # Instantiation of a
            # Porter stemmer
            self.en_stop_words = stopwords.words("english") # Get and
            # English stop word list

    def clean_data(self):
        """Method to clean the data

        It cleans all the attributes.
        """
        for i, bug_report in enumerate(self.json_data):
            print(i)
            bug_report["bug_id"] = self \
            .__clean_bug_id(bug_report["bug_id"])
            bug_report["title"] = self \
            .__clean_title(bug_report["title"])
            bug_report["description"] = self \
            .__clean_description(bug_report["description"])

    def __clean_title(self, title):
        title = "" if title is None else title
        title = self.html_parser.unescape(title)
        tokens = word_tokenize(title)
        tokens = [self.stemmer.stem(token) for token in tokens]
        new_tokens = [token for token in tokens if token.lower() \
        not in self.en_stop_words]
        return [token for token in new_tokens \
        if not all(char in string.punctuation for char in token)]

    def __clean_description(self, description):
        description = "" if description is None else description
        description = self.html_parser.unescape(description)
        description = re.sub(r"\n+", " ", description)
        description = re.sub(r"\r+", " ", description)
        description = re.sub(r"\t+", " ", description)
        tokens = word_tokenize(description)
        tokens = [self.stemmer.stem(token) for token in tokens]
        new_tokens = [token for token in tokens if token.lower() \
        not in self.en_stop_words]
        return [token for token in new_tokens \
        if not all(char in string.punctuation for char in token)]

    def __clean_bug_id(self, id):
        id = "" if id is None else id
        id = self.html_parser.unescape(id)
        return id.replace(u"Bug\u00a0", "")

    def write_data(self, data_file):
        """Method to write the data

        It writes the data into a JSON file which path is given in
        parameter.
        """
        with open(data_file, "w") as ouput_file:
            json.dump(self.json_data, ouput_file)

if __name__ == "__main__":
    """Main program"""
    input_data_file = "../scrap_eclipse_jdt/brs.json" # Path of the
    # file containing the data
    output_data_file = "./pre_processed_output.json" # Path of the
    # output file
    # Instantiation of the Pre-processer
    data_pre_processer = DataPreProcesser(input_data_file)
    data_pre_processer.clean_data() # We clean the data
    data_pre_processer.write_data(output_data_file) # We write the
    # data into the given output file
