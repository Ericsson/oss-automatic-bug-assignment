# -*- coding: utf-8 -*-
"""
.. module:: experiment_3_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to plot the 
              results related to the third experiment of the thesis. 
              The experiment consists mainly of comparing several 
              feature selection techniques and selecting the best 
              one.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir) 
from results_plotter import ResultsPlotter

class Experiment3ResultsPlotter(ResultsPlotter):
    
    # Below, there is a dictionary which maps each key to a smaller 
    # one for readability purposes
    KEY_MAPPING = {
        "GridSearch chi2 Linear SVM": "CHI-2",
        "GridSearch f_classif Linear SVM": "ANOVA",
        "GridSearch mutual_info_classif Linear SVM": "MI",
        "RFECV SVM": "RFE"
    }

    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        super().__init__(cleaned_results_file_name)
        
    @classmethod
    def get_cleaned_list_from_dict(cls, dict_content):
        """Cleans the results related to a specific metric
    
        A dictionary containing the results related to a specific 
        metric is given in parameter. The aforementioned dictionary 
        should only contain the values of a specific metric, related 
        to the different feature selection configurations which are 
        compared. A list of tuples is then generated. For each 
        configuration of each technique, a tuple is added to this 
        list. This tuple contains both a unique key designating the 
        configuration and its value for the specific metric. The 
        generated list is eventually returned.
        
        :param dict_content: The values of a metric, related to the
        different selected feature selection configurations which are
        compared.
        :type dict_content: dictionary.
        :returns:  list -- the list of tuples containing the unique 
        keys and the metric values related to the different feature 
        selection configurations.
        :raises: KeyError        
        """
        generated_list = []
        for key, value in dict_content.items():
            if key == "GridSearch var_threshold Linear SVM":
                continue
            if key == "RFECV SVM":
                i = 0
                for i in range(5):
                    generated_list.append((cls.get_small_key(key) + \
                        "-{}".format(10+i*20), value[2+i*2]))
            else:
                i = 0
                for accuracy in value:
                    generated_list.append((cls.get_small_key(key) + \
                        "-{}".format(10+i*20), accuracy))
                    i += 1
        return generated_list
    
    @classmethod
    def get_small_key(cls, key):
        """Maps a key to a smaller one and returns it.
        
        The original key was used to store the results related to a 
        specific technique ("GridSearch chi2 Linear SVM", "GridSearch 
        f_classif Linear SVM", "GridSearch mutual_info_classif Linear 
        SVM" or "RFECV SVM"). This key is then mapped to an acronym 
        for readability purposes. This acronym is eventually returned.
        
        :param key: A key used to designate a technique.
        :type key: string.
        :returns:  string -- the acronym
        :raises: KeyError
        """
        return cls.KEY_MAPPING[key]
    
    @abc.abstractmethod
    def plot_results(self):
        """Plots the results in horizontal bar chart(s)
        
        This method plots the results and saves them into PNG files.
        """
        super().plot_results()
        
        for plot_parameter in self.plot_parameters:
            # The dictionary below contains the values of a metric 
            # related to the different configurations
            dict_of_metric_values = self. \
            _cleaned_results[plot_parameter["key"]]
            
            # Below, we generate a list of tuples based on the above
            # dictionary. The first element of each tuple is the 
            # reduced form of the original key. The second element is
            # the metric values.
            list_of_metric_values = self. \
            get_cleaned_list_from_dict(dict_of_metric_values)
            
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_metric_values.sort(key=lambda x: x[1], \
                                       reverse=True)
            
            self.print_dict(plot_parameter["debug_title"], \
                            list_of_metric_values) # Debug
            
            # Below, we generate two lists from the above mentioned 
            # list of tuples
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