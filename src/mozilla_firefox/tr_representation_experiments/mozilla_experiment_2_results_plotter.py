# -*- coding: utf-8 -*-
"""
.. module:: mozilla_experiment_2_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains a class used to plot the results 
              related to the second experiment of the thesis conducted
              on the bug reports of Mozilla Firefox. The experiment 
              consists mainly of comparing several feature extraction
              techniques and selecting the best one.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from tr_representation_experiments.experiment_2_results_plotter \
import Experiment2ResultsPlotter

class MozillaExperiment2ResultsPlotter(Experiment2ResultsPlotter):
    
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(cleaned_results_file_name)
        
    def plot_results(self):
        """This method plots the results in chart(s)"""
        self.plot_parameters = [
            {
                "key": "not_combined_avg_accuracy",
                "x_lim_min": 0.11,
                "x_lim_max": 0.176,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "title_font_size": 45,
                "labels_font_size": 40, 
                "y_tick_labels_font_size": 35,
                "title": "Accuracy of the different feature " + \
                "extraction techniques (without combination of " + \
                "features)",
                "file_name" : "experiment_21.png",
                "debug_title": "Average Accuracy (Without Combination)",
                "combined": False,
                "bars_labels_space": 0.0005
            },
            {
                "key": "combined_avg_accuracy",
                "x_lim_min": 0.124,
                "x_lim_max": 0.174,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "title_font_size": 45,
                "labels_font_size": 40, 
                "y_tick_labels_font_size": 35,
                "title": "Accuracy of the different feature " + \
                "extraction techniques (with combination of " + \
                "features)",
                "file_name": "experiment_22.png",
                "debug_title": "Average Accuracy (With Combination)",
                "combined": True,
                "bars_labels_space": 0.0005
            },
            {
                "key": "not_combined_avg_mrr",
                "x_lim_min": 0.28,
                "x_lim_max": 0.355,
                "x_label": "MRR",
                "y_label": "Configurations",
                "title_font_size": 45,
                "labels_font_size": 40, 
                "y_tick_labels_font_size": 35,
                "title": "MRR of the different feature " + \
                "extraction techniques (without combination of " + \
                "features)",
                "file_name" : "experiment_23.png",
                "debug_title": "Average MRR (Without Combination)",
                "combined": False,
                "bars_labels_space": 0.0005
            },
            {
                "key": "combined_avg_mrr",
                "x_lim_min": 0.29,
                "x_lim_max": 0.354,
                "x_label": "MRR",
                "y_label": "Configurations",
                "title_font_size": 45,
                "labels_font_size": 40, 
                "y_tick_labels_font_size": 35,
                "title": "MRR of the different feature " + \
                "extraction techniques (with combination of " + \
                "features)",
                "file_name": "experiment_24.png",
                "debug_title": "Average MRR (With Combination)",
                "combined": True,
                "bars_labels_space": 0.0005
            }   
        ]
        super().plot_results()

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_tr_representation_" + \
    "experiment_results.json"
    experiment_2_results_plotter = \
    MozillaExperiment2ResultsPlotter(cleaned_results_file_name)
    experiment_2_results_plotter.plot_results()

if __name__ == "__main__":
    main()