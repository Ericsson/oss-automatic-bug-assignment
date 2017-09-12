# -*- coding: utf-8 -*-
"""
.. module:: experiment_1_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to plot the 
              results related to the first experiment of the thesis. 
              The experiment consists mainly of comparing several 
              combinations of pre-processing techniques and selecting 
              the best one.

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

class Experiment1ResultsPlotter(ResultsPlotter):
    
    # Below, there is a list of tuples to map the names of the 
    # different pre-processing steps to their associated acronyms
    ACRONYMS_MAPPING = [ \
        (1, "C"), \
        (3, "S"), \
        (5, "L"), \
        (7, "SW"), \
        (11, "P"), \
        (14, "N"), \
        (17, "LC")
    ]
    
    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        super().__init__(cleaned_results_file_name)
    
    @classmethod
    def get_cleaned_list_from_dict(cls, dict_content):
        """Returns a cleaned list from a dictionary"""
        generated_list = []
        for key, value in dict_content.items():
            generated_list.append((cls.get_small_name(key), value))
        return generated_list
    
    @classmethod
    def get_small_name(cls, name_of_file):
        """Return the reduced form of a file name"""
        name_of_file_parts = name_of_file.split(".")
        name_of_file_without_extension = name_of_file_parts[0]
        name_of_file_parts = name_of_file_without_extension.split("_")
        small_name = ""
        # print(name_of_file) # Debug
        id = 1
        param_number = 5
        for index, acronym in cls.ACRONYMS_MAPPING:
            # print(acronym) # Debug
            # print(name_of_file_parts[index]) # Debug
            if index > 1:
                small_name += "|"
            if param_number == 5 and name_of_file_parts[index] ==  "with":
                id += 3 * 2**4
            elif param_number == 4:
                if name_of_file_parts[5] ==  "with":
                    id += 2**4
                elif name_of_file_parts[index] ==  "with":
                    id += 2 * 2**4
            elif param_number < 3 and name_of_file_parts[index] ==  "with":
                id += 2 * 2**param_number
            small_name += acronym if name_of_file_parts[index] ==  "with" else "NOT({})".format(acronym)
            param_number -= 1
        small_name += " #1." + "{:<2d}".format(int(id))
        return small_name
    
    @abc.abstractmethod
    def plot_results(self):
        """This method plots the results in chart(s)"""
        super().plot_results()
        fig_width_inches = 25
        fig_height_inches = 72
        for plot_parameter in self.plot_parameters:
            list_of_metric_values = []
            for key, value in self \
            ._cleaned_results[plot_parameter["key"]].items():
                list_of_metric_values.append((key, value))
        
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_metric_values.sort(key=lambda x: x[1], reverse=True)
        
            print("Number of configurations: {}" \
                  .format(len(list_of_metric_values))) # Debug
            
            ResultsPlotter.print_dict(
            plot_parameter["x_label"], \
            list_of_metric_values) # Debug
        
            # Below, we generate a list of tuples based on the above
            # dictionary. The first element of each tuple is the 
            # reduced form of the associated file name. The second 
            # element is the metric value.
            list_of_metric_values = self.get_cleaned_list_from_dict( \
            self._cleaned_results[plot_parameter["key"]])
        
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_metric_values.sort(key=lambda x: x[1], \
                                       reverse=True)
    
            self.print_dict(plot_parameter["debug_title"], \
                            list_of_metric_values) # Debug
        
            # Below, we generate two lists from the above mentioned list of 
            # tuples 
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
                          y_tick_labels_font_size, \
                          bars_labels_space, fig_width_inches, \
                          fig_height_inches)