# -*- coding: utf-8 -*-
"""
.. module:: results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to plot the 
              results related to all the experiments of the thesis.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import matplotlib.pyplot as plt
import abc
import os

from utilities import load_json_file

class ResultsPlotter(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        self._cleaned_results_file_name = os.path.join( \
        self._current_dir, cleaned_results_file_name)
        self._cleaned_results = \
        self.load_cleaned_results()
    
    def load_cleaned_results(self):
        """Opens and loads the cleaned results from a JSON file."""
        return load_json_file(self._cleaned_results_file_name)
    
    @staticmethod
    def print_dict(title, dictionary):
        """Prints the content of a dictionary for debugging"""
        print("\n### {} ###".format(title))
        for acronym, metric_value in dictionary:
            print("{}: {}".format(acronym, metric_value)) # Debug
            
    @staticmethod
    def generate_two_lists(list_of_two_elements_tuples):
        """Returns two lists from a list of tuples"""
        list_1 = []
        list_2 = []
        for element_1, element_2 in list_of_two_elements_tuples:
            list_1.append(element_1)
            list_2.append(element_2)
        return list_1, list_2
            
    @staticmethod
    def add_labels_to_bars(x_values, ax, bars_labels_space, \
                           labels_font_size):
        """Add labels (percentage) to some given bars"""
        for i, v in enumerate(x_values):
            ax.text(v + bars_labels_space, i+1, "{:.2%}".format(v), \
                    color='k', fontsize=labels_font_size, \
                    verticalalignment='center')
    
    @abc.abstractmethod
    def plot_results(self):
        """This method plots the results in chart(s)"""
        pass
    
    def plot_bar(self, y_labels, x_values, x_lim_min, x_lim_max, \
                 x_label, y_label, title, file_name=None, \
                 labels_font_size=35, y_tick_labels_font_size=20, 
                 bars_labels_space=0.005, fig_width_inches=25, \
                 fig_height_inches=40):
        """Makes a bar plot"""
        # We get a sub-plot inside a figure and its associated axis
        fig, ax = plt.subplots()
    
        # We set the size of the figure
        fig.set_size_inches(fig_width_inches, fig_height_inches)
    
        height = 0.9 # Height of each bin
    
        # Below, we compute the position of each bin
        bins = list(map(lambda y: y, range(1, len(y_labels)+1)))
    
        plt.title(title, fontsize=40)
    
        # We set the x and y limits of the chart
        plt.ylim([1-height, len(y_labels) + height])
        plt.xlim([x_lim_min, x_lim_max])
    
        # We set the labels of the chart
        plt.ylabel(y_label, fontsize=labels_font_size)
        plt.xlabel(x_label, fontsize=labels_font_size)
        
        # We make the bar plot
        ax.barh(bins, x_values, height=height, alpha=0.8)
        
        # We set the x-ticks and the labels of the x-ticks 
        ax.set_yticks(list(map(lambda y: y, range(1,len(y_labels)+1))))
        ax.set_yticklabels(y_labels, fontsize=y_tick_labels_font_size)
    
        self.add_labels_to_bars(x_values, ax, bars_labels_space, \
                                y_tick_labels_font_size)
    
        save_file = None if not file_name \
        else os.path.join(self._current_dir, file_name)  
    
        if save_file:
            # bbox_inches="tight"
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()