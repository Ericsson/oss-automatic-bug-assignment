# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
import time
import numpy as np
import os
import inspect
import logging
import csv
import sys
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from experiment import Experiment
from data_pre_processer import DataPreProcesser
from utilities import print_log

class PreProcessingExperiment(Experiment):

    @abc.abstractmethod
    def __init__(self, developers_dict_file, \
    developers_list_file, clean_brs=False, use_stemmer=False, \
    use_lemmatizer=False, stop_words_removal=False, \
    punctuation_removal=False, numbers_removal=False):
        """Constructor"""
        super().__init__(developers_dict_file, developers_list_file)
        # Used to store the path of the JSON file containing the
        # sorted bug reports
        self.data_file = None 
        
        # TO DO: Management of the reference to the data pre-processer
        
        # Below, we set a default configuration
        self.set_config(clean_brs=clean_brs, \
        use_stemmer=use_stemmer, use_lemmatizer=use_lemmatizer, \
        stop_words_removal=stop_words_removal, \
        punctuation_removal=punctuation_removal, \
        numbers_removal=numbers_removal)
        
        self._data_pre_processer = None # Used to store a reference 
        # to a data pre-processer
        
        self._model = LinearSVC(random_state=0) # The model used
        
        # Below, there is a dictionary to store the accuracies (per 
        # fold) of each configuration
        self._configurations_accuracies = {}
        
        # Below, there is a dictionary to store the MRR values (per 
        # fold) of each configuration
        self._configurations_mrr_values = {}
        
        self._cleaned_results_file_name = "cleaned_pre_" + \
        "processing_experiment_results.json"
        
    def set_config(self, clean_brs, use_stemmer, use_lemmatizer, \
    stop_words_removal, punctuation_removal, numbers_removal):
        self.clean_brs = clean_brs
        self.use_stemmer = use_stemmer
        self.use_lemmatizer = use_lemmatizer
        self.stop_words_removal = stop_words_removal
        self.punctuation_removal = punctuation_removal
        self.numbers_removal = numbers_removal

    @property
    def data_file(self):
        return self._data_file 
    
    @data_file.setter
    def data_file(self, data_file):
        self._data_file = data_file

    @property
    def clean_brs(self):
        return self._clean_brs 
        
    @clean_brs.setter
    def clean_brs(self, clean_brs):
        self._clean_brs = clean_brs

    @property
    def use_stemmer(self):
        return self._use_stemmer 
        
    @use_stemmer.setter
    def use_stemmer(self, use_stemmer):
        self._use_stemmer = use_stemmer
        
    @property
    def use_lemmatizer(self):
        return self._use_lemmatizer 
        
    @use_lemmatizer.setter
    def use_lemmatizer(self, use_lemmatizer):
        self._use_lemmatizer = use_lemmatizer
        
    @property
    def stop_words_removal(self):
        return self._stop_words_removal 
        
    @stop_words_removal.setter
    def stop_words_removal(self, stop_words_removal):
        self._stop_words_removal = stop_words_removal
        
    @property
    def punctuation_removal(self):
        return self._punctuation_removal 
        
    @punctuation_removal.setter
    def punctuation_removal(self, punctuation_removal):
        self._punctuation_removal = punctuation_removal
        
    @property
    def numbers_removal(self):
        return self._numbers_removal 
        
    @numbers_removal.setter
    def numbers_removal(self, numbers_removal):
        self._numbers_removal = numbers_removal

    def get_file_name(self):
        # Below, we are giving a relevant name to the output file
        clean_brs_string = "" if self.clean_brs else "out" 
        use_stemmer_string = "" if self.use_stemmer else "out"
        use_lemmatizer_string = "" if self.use_lemmatizer else "out"
        stop_words_removal_string = "" if self.stop_words_removal else "out"
        punctuation_removal_string = "" if self.punctuation_removal else "out"
        numbers_removal_string = "" if self.numbers_removal else "out"   
        output_data_file = \
        "output_with{}_cleaning_" \
        .format(clean_brs_string) + \
        "with{}_stemming_" \
        .format(use_stemmer_string) + \
        "with{}_lemmatizing_" \
        .format(use_lemmatizer_string) + \
        "with{}_stop_words_removal_" \
        .format(stop_words_removal_string) + \
        "with{}_punctuation_removal_" \
        .format(punctuation_removal_string) + \
        "with{}_numbers_removal.json" \
        .format(numbers_removal_string)    
        return output_data_file
    
    def generate_output_file(self):
        start_time = time.time() # We get the time expressed in 
        # seconds since the epoch
        output_data_file = self.get_file_name()
        print("Generating {}...".format(output_data_file))
        print(self._data_file)
        self._data_pre_processer = DataPreProcesser(self._data_file, \
        clean_brs=self._clean_brs, use_stemmer=self._use_stemmer, \
        use_lemmatizer=self._use_lemmatizer, \
        stop_words_removal=self._stop_words_removal, \
        punctuation_removal=self._punctuation_removal, \
        numbers_removal=self._numbers_removal) # Instantiation of the 
        # Pre-processer
        output_data_file = os.path.join(self._current_dir, \
        output_data_file)
        self._data_pre_processer.clean_data() # We clean the data
        self._data_pre_processer.write_data(output_data_file) # We 
        # write the data into the given output file
        del self._data_pre_processer
        print("--- {} seconds ---".format(time.time() - start_time))
    
    def generate_all_output_files(self):
        for clean_brs in [False, True]:
            for use_stemmer, use_lemmatizer in [(False, False), \
            (False, True), (True, False)]:
                for stop_words_removal in [False, True]:
                    for punctuation_removal in [False, True]:
                        for numbers_removal in [False, True]:
                            self.set_config(clean_brs=clean_brs, \
                            use_stemmer=use_stemmer, \
                            use_lemmatizer=use_lemmatizer, \
                            stop_words_removal=stop_words_removal, \
                            punctuation_removal=punctuation_removal, \
                            numbers_removal=numbers_removal)
                            self.generate_output_file()
                                
    def train_predict_all_output_files(self):
        to_lower_case = None # Used to manage the lower case 
        # parameter in the various configurations
        j = 1 # Iterator
        data_set_file_list = None # Temporary variable used to build 
        # the name of each configuration
        with_lower_case = "_with_to_lower_case." # Variable used to 
        # build the name of each configuration using conversion to 
        # lower case
        without_lower_case = "_without_to_lower_case." # Variable used
        # to build the name of each configuration not using conversion
        # to lower case
        temp = None  # Temporary variable used to build the name of 
        # each configuration
        print(os.listdir(self._current_dir)) # Debug
        print(len(os.listdir(self._current_dir))) # Debug
        for file in os.listdir(self._current_dir):
#             if j == 2:
#                 break
            print(file) # Debug
            if file.endswith(".json") and \
            file != "cleaned_pre_processing_experiment_results.json":
                # print(os.path.join(self._current_dir, file)) # Debug
                start_time = time.time() # We get the time expressed 
                # in seconds since the epoch
                self._data_set_file = file
                np.random.seed(0) # We set the seed
                self._build_data_set() # We build the data set
                for to_lower_case in [False, True]: 
                    # We iterate to manage the potential conversion 
                    # to lower case 
                    temp = self._data_set_file
                    data_set_file_list = self._data_set_file.split(".")
                    if to_lower_case:
                        self._data_set_file = \
                        data_set_file_list[0] + with_lower_case + \
                        data_set_file_list[1] 
                    else:
                        self._data_set_file = \
                        data_set_file_list[0] + without_lower_case + \
                        data_set_file_list[1]
                    print_log("##### File name: {} #####" \
                              .format(self._data_set_file)) # Debug
                    print_log("--- {} seconds ---" \
                              .format(time.time() - start_time))
                    i = 1                    
                    for train_indices, val_indices in self._tscv.split(self._train_set):   
                        print_log("********* Evaluation on fold {} *********"\
                        .format(i)) # Debug
                        
                        print_log("We count the occurrence of each term") # Debug
                        count_vectorizer = CountVectorizer(lowercase=to_lower_case, \
                        token_pattern=u'(?u)\S+')
                        X_counts = count_vectorizer \
                        .fit_transform(self._train_set.iloc[train_indices]['text'].values)
                        
                        print_log("Use of the TF-IDF model") # Debug
                        tfidf_transformer = TfidfTransformer(use_idf=False, \
                        smooth_idf=False)
                        print_log(X_counts.shape) # Debug
                        
                        print_log("Computation of the weights of the TF-IDF model")
                        X_train = tfidf_transformer.fit_transform(X_counts)
                        y_train = self._train_set.iloc[train_indices]['class'].values
                        print_log(X_train.shape)
                        
                        print_log("--- {} seconds ---".format(time.time() - start_time))
                        
                        print_log("Training of the models") # Debug
                        self._model.fit(X_train, y_train)
                        
                        print_log("--- {} seconds ---".format(time.time() - start_time))
                            
                        print_log("We count the occurrence of " + \
                                  "each term in the val. set") # Debug
                        X_val_counts = count_vectorizer \
                        .transform(self._train_set.iloc[val_indices]['text'].values)
                        print_log("Computation of the weights of " + \
                                  "the TF-IDF model for the " + \
                                  "validation set") # Debug
                        X_val = tfidf_transformer. \
                        transform(X_val_counts)
                        y_val = self._train_set. \
                        iloc[val_indices]['class'].values
                        print_log("Making predictions") # Debug
                        
                        if i == 1:
                            self._configurations_accuracies[self._data_set_file] = []
                            self._configurations_mrr_values[self._data_set_file] = []
                        self._configurations_accuracies[self._data_set_file].append(\
                        np.mean(self._model.predict(X_val) == y_val))

                        found_function = False                        
                        try:
                            if callable(getattr(self._model, "predict_proba")):
#                                 print_log(self._model.classes_)
#                                 print_log(self._model.predict_proba(X_val))
                                lb = LabelBinarizer()        
                                _ = lb.fit_transform(self._model.classes_)
#                                 print_log(lb.classes_)
#                                 print_log(y_classes_bin)
#                                 print_log(lb.transform(["X"]))  
                                y_val_bin = lb.transform(y_val)
                                self._configurations_mrr_values[self._data_set_file].append(\
                                label_ranking_average_precision_score( \
                                y_val_bin, \
                                self._model.predict_proba(X_val)))
                                found_function = True
                        except AttributeError:
                            pass
                        
                        try:
                            if not found_function and callable(getattr(self._model, "decision_function")):
#                                 print_log(self._model.classes_)
#                                 print_log(self._model.decision_function(X_val))
                                lb = LabelBinarizer()        
                                _ = lb.fit_transform(self._model.classes_)
#                                 print_log(lb.classes_)
#                                 print_log(y_classes_bin)
#                                 print_log(lb.transform(["X"]))  
                                y_val_bin = lb.transform(y_val)
                                self._configurations_mrr_values[self._data_set_file].append(\
                                label_ranking_average_precision_score( \
                                y_val_bin, \
                                self._model.decision_function(X_val)))
                                found_function = True
                        except AttributeError:
                            pass
                        print_log("Mean Reciprocal Rank:")
                        print_log(self._configurations_mrr_values[self._data_set_file][-1])                        
                        print_log("--- {} seconds ---".format(time.time() - start_time))
                        
                        i += 1
                        
                    self._data_set_file = temp
                   
                print_log("*** File {} done ***".format(j)) # Debug
                j += 1
        
        self._results_to_save_to_a_file["avg_accuracy"] = {}
        self._results_to_save_to_a_file["avg_mrr"] = {}
        
        avg_accuracy = None
        avg_mrr = None
        
        # Below, we print the average accuracies
        for key, value in self._configurations_accuracies.items():
            print_log("Accuracy of {}".format(key)) # Debug
            print_log("Each fold")
            print_log(value)
            print_log("Average")
            avg_accuracy = sum(value)/len(value) 
            self._results_to_save_to_a_file["avg_accuracy"][key] = \
            avg_accuracy
            print_log(avg_accuracy) # Debug
            print_log("MRR of {}".format(key)) # Debug
            print_log("Each fold")
            mrr = self._configurations_mrr_values[key]
            print_log(mrr)
            print_log("Average")
            avg_mrr = sum(mrr)/len(mrr) 
            self._results_to_save_to_a_file["avg_mrr"][key] = \
            avg_mrr
            print_log(avg_mrr)
            
        self._results_to_save_to_a_file["accuracy_per_fold"] = \
        self._configurations_accuracies
        self._results_to_save_to_a_file["mrr_per_fold"] = \
        self._configurations_mrr_values
        print_log("--- {} seconds ---".format(time.time() - start_time))
        self._train_set = None
        self._test_set = None
        self._configurations_accuracies = {}
        self._configurations_mrr_values = {}
        
    def conduct_experiment(self):
        """Method used to conduct the experiment"""
        self.generate_all_output_files()
        self.train_predict_all_output_files()
        super().conduct_experiment()