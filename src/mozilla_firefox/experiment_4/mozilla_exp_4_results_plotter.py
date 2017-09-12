# -*- coding: utf-8 -*-
"""
.. module:: mozilla_experiment_4_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains a class used to plot the results 
              related to the last experiment of the thesis conducted 
              on the bug reports of Mozilla Firefox. The experiment 
              consists mainly of tuning several classifiers and 
              selecting the best performing one.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from experiment_4.exp_4_results_plotter import Exp4ResultsPlotter

class MozillaExp4ResultsPlotter(Exp4ResultsPlotter):
    
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(cleaned_results_file_name)
        
    def plot_results(self):
        """This method plots the results in chart(s)"""
        self.plot_parameters = [
            {
                "key": "normal_avg",
                "x_lim_min": [0.04, 0.20],
                "x_lim_max": [0.1555, 0.35],
                "x_label": ["Accuracy", "MRR"],
                "y_label": ["Configurations", "Configurations"],
                "titles_font_size": [60, 60],
                "labels_font_size": [51, 51], 
                "y_tick_labels_font_size": [42, 42],
                "bars_labels_space": [0.001, 0.001], 
                "title": ["Accuracy of the different " + \
                          "configurations (grid search)",
                          "MRR of the different configurations " + \
                          "(grid search)"],
                "file_name" : ["experiment_41.png", \
                               "experiment_42.png"],
                "debug_title": ["Average Accuracy (Grid Search)",
                                "Average MRR (Grid Search)"] 
            },
            {
                "key": "random_avg",
                "x_lim_min": [0.04, 0.20],
                "x_lim_max": [0.1565, 0.353],
                "x_label": ["Accuracy", "MRR"],
                "y_label": ["Configurations", "Configurations"],
                "titles_font_size": [60, 60],
                "labels_font_size": [51, 51], 
                "y_tick_labels_font_size": [42, 42],
                "bars_labels_space": [0.001, 0.001],
                "title": ["Accuracy of the different " + \
                          "configurations (random search)",
                          "MRR of the different configurations " + \
                          "(random search)"],
                "file_name" : ["experiment_43.png", \
                               "experiment_44.png"],
                "debug_title": ["Average Accuracy (Random Search)",
                                "Average MRR (Random Search)"] 
            }       
        ]
        super().plot_results()

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_tuning_individual_" + \
    "classifier_generic_experiment_results.json"
    mozilla_exp_4_results_plotter = \
    MozillaExp4ResultsPlotter(cleaned_results_file_name)
    mozilla_exp_4_results_plotter.plot_results()

if __name__ == "__main__":
    main()