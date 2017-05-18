# -*- coding: utf-8 -*-
"""
.. module:: eclipse_experiment_1_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains a class used to plot the results 
              related to the first experiment of the thesis conducted 
              on the bug reports of Eclipse JDT. The experiment 
              consists mainly of comparing several combinations of 
              pre-processing techniques and selecting the best one.

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
from pre_processing_experiments.experiment_1_results_plotter \
import Experiment1ResultsPlotter

class EclipseExperiment1ResultsPlotter(Experiment1ResultsPlotter):
    
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(cleaned_results_file_name)

    def plot_results(self):
        """This method plots the results in chart(s)"""
        self.plot_parameters = [
            {
                "key": "avg_accuracy",
                "x_lim_min": 0.25,
                "x_lim_max": 0.277,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 35,
                "title": "Accuracy of the different pre-processing " \
                "configurations",
                "file_name" : "experiment_11.png",
                "debug_title": "Average Accuracy",
                "bars_labels_space": 0.0003,
                "fig_width_inches": 25,
                "fig_height_inches": 60 
            },
            {
                "key": "avg_mrr",
                "x_lim_min": 0.405,
                "x_lim_max": 0.429,
                "x_label": "MRR",
                "y_label": "Configurations",
                "labels_font_size": 35,
                "y_tick_labels_font_size": 35,
                "title": "MRR of the different pre-processing " \
                "configurations",
                "file_name": "experiment_12.png",
                "debug_title": "Average MRR",
                "bars_labels_space": 0.0003,
                "fig_width_inches": 25,
                "fig_height_inches": 60 
            }       
        ]
        super().plot_results()

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_pre_processing_" + \
    "experiment_results.json"
    experiment_1_results_plotter = \
    EclipseExperiment1ResultsPlotter(cleaned_results_file_name)
    experiment_1_results_plotter.plot_results()

if __name__ == "__main__":
    main()