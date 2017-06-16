# -*- coding: utf-8 -*-
"""
.. module:: experiment_2_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to plot the 
              results related to the second experiment of the thesis. 
              The experiment consists mainly of comparing several 
              feature extraction techniques and selecting the best 
              one.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from results_plotter import ResultsPlotter

class Experiment2ResultsPlotter(ResultsPlotter):

    # Below, there is a dictionary which map each key to a smaller one 
    # for readability purposes
    KEY_MAPPING = {
        "Boolean SVM": "BOOL",
        "TF SVM": "TF",
        "TF IDF SVM": "TF-IDF",
        "GridSearch Boolean Truncated SVD SVM": "BOOL+LSI",
        "GridSearch TF Truncated SVD SVM": "TF+LSI",
        "GridSearch TF IDF Truncated SVD SVM": "TF-IDF+LSI",
        "GridSearch Boolean NMF SVM": "BOOL+NMF",
        "GridSearch TF NMF SVM": "TF+NMF",
        "GridSearch TF IDF NMF SVM": "TF-IDF+NMF",
    }
    
    CONFS_1 = [
        "BOOL",
        "TF",
        "TF-IDF",
        "BOOL+LSI-10",
        "BOOL+LSI-30",
        "BOOL+LSI-50",
        "BOOL+LSI-70",
        "BOOL+LSI-90",
        "TF+LSI-10",
        "TF+LSI-30",
        "TF+LSI-50",
        "TF+LSI-70",
        "TF+LSI-90",
        "TF-IDF+LSI-10",
        "TF-IDF+LSI-30",
        "TF-IDF+LSI-50",
        "TF-IDF+LSI-70",
        "TF-IDF+LSI-90",
        "BOOL+NMF-10",
        "BOOL+NMF-30",
        "BOOL+NMF-50",
        "BOOL+NMF-70",
        "BOOL+NMF-90",
        "TF+NMF-10",
        "TF+NMF-30",
        "TF+NMF-50",
        "TF+NMF-70",
        "TF+NMF-90",
        "TF-IDF+NMF-10",
        "TF-IDF+NMF-30",
        "TF-IDF+NMF-50",
        "TF-IDF+NMF-70",
        "TF-IDF+NMF-90"
    ]
    
    CONFS_2 = [
        "BOOL&TF",
        "BOOL&TF-IDF",
        "BOOL&BOOL+LSI",
        "BOOL&TF+LSI",
        "BOOL&TF-IDF+LSI",
        "BOOL&BOOL+NMF",
        "BOOL&TF+NMF",
        "BOOL&TF-IDF+NMF",
        "TF&TF-IDF",
        "TF&BOOL+LSI",
        "TF&TF+LSI",
        "TF&TF-IDF+LSI",
        "TF&BOOL+NMF",
        "TF&TF+NMF",
        "TF&TF-IDF+NMF",
        "TF-IDF&BOOL+LSI",
        "TF-IDF&TF+LSI",
        "TF-IDF&TF-IDF+LSI",
        "TF-IDF&BOOL+NMF",
        "TF-IDF&TF+NMF",
        "TF-IDF&TF-IDF+NMF",
        "BOOL+LSI&TF+LSI",
        "BOOL+LSI&TF-IDF+LSI",
        "BOOL+LSI&BOOL+NMF",
        "BOOL+LSI&TF+NMF",
        "BOOL+LSI&TF-IDF+NMF",
        "TF+LSI&TF-IDF+LSI",
        "TF+LSI&BOOL+NMF",
        "TF+LSI&TF+NMF",
        "TF+LSI&TF-IDF+NMF",
        "TF-IDF+LSI&BOOL+NMF",
        "TF-IDF+LSI&TF+NMF",
        "TF-IDF+LSI&TF-IDF+NMF",
        "BOOL+NMF&TF+NMF",
        "BOOL+NMF&TF-IDF+NMF",
        "TF+NMF&TF-IDF+NMF"
    ]
    
    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        super().__init__(cleaned_results_file_name)
    
    @classmethod
    def get_cleaned_list_from_dict_without_combination(cls, dict_content):
        """Returns a cleaned list from a dictionary
    
        The aforementioned dictionary should only contain the metric 
        values related to different feature extraction techniques 
        (without combination)
        """
        generated_list = []
        conf = ""
        for key, value in dict_content.items():
            if isinstance(value, list):
                i = 0
                for accuracy in value:
                    conf = cls.get_small_key(key) + \
                    "-{}".format(10+i*20)
                    conf += " #3.1." + "{:<2d}" \
                    .format(cls.CONFS_1.index(conf) + 1)
                    generated_list.append((conf, accuracy))
                    i += 1
            else:
                conf = cls.get_small_key(key)
                conf += " #3.1." + "{:<2d}" \
                .format(cls.CONFS_1.index(conf) + 1)
                generated_list.append((conf, value))        
        return generated_list
    
    @classmethod
    def get_cleaned_list_from_dict_with_combination(cls, \
                                                    dict_content):
        """Returns a cleaned list from a dictionary
    
        The aforementioned dictionary should only contain the metric 
        values related to different feature extraction techniques 
        (with combination)
        """
        generated_list = []
        for key, value in dict_content.items():
            key_split = key.split("_")
            conf = cls.get_small_key(key_split[0]) + "&" + \
            cls.get_small_key(key_split[1])
            conf += " #3.2." + "{:<2d}" \
            .format(cls.CONFS_2.index(conf) + 1)
            generated_list.append((conf, \
                                   value))
        return generated_list
    
    @classmethod
    def get_small_key(cls, key):
        """Return the reduced form of a key"""
        return cls.KEY_MAPPING[key]

    @abc.abstractmethod
    def plot_results(self):
        """This method plots the results in chart(s)"""
        super().plot_results()
        
        for plot_parameter in self.plot_parameters:
            # The dictionary below contains the values of a metric 
            # related to the different representations
            dict_of_metric_values = self. \
            _cleaned_results[plot_parameter["key"]]
            
            # Below, we generate a list of tuples based on the above 
            # dictionary. The first element of each tuple is the 
            # reduced form of the original key. The second element is 
            # the metric value.
            if plot_parameter["combined"]:
                list_of_metric_values = \
                self.get_cleaned_list_from_dict_with_combination( \
                dict_of_metric_values)

            else:
                list_of_metric_values = \
                self.get_cleaned_list_from_dict_without_combination( \
                dict_of_metric_values)
    
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_metric_values.sort(key=lambda x: x[1], \
                                       reverse=True)
    
            self.print_dict(plot_parameter["debug_title"], \
                            list_of_metric_values) # Debug

    
            # Below, we generate two lists from the first above mentioned list
            # of tuples
            y_labels, x_values = self. \
            generate_two_lists(list_of_metric_values)
            
            x_lim_min = plot_parameter["x_lim_min"]
            x_lim_max = plot_parameter["x_lim_max"]
            x_label = plot_parameter["x_label"]
            y_label = plot_parameter["y_label"]
            title_font_size = plot_parameter["title_font_size"]
            labels_font_size = plot_parameter["labels_font_size"]
            y_tick_labels_font_size = \
            plot_parameter["y_tick_labels_font_size"]
            title = plot_parameter["title"]
            file_name = plot_parameter["file_name"]
            bars_labels_space = plot_parameter["bars_labels_space"]
            self.plot_bar(y_labels, x_values, x_lim_min, x_lim_max, \
                          x_label, y_label, title, file_name, \
                          title_font_size, labels_font_size, \
                          y_tick_labels_font_size, bars_labels_space)