# -*- coding: utf-8 -*-
"""
.. module:: mozilla_experiment_3_results_plotter
   :platform: Unix, Windows
   :synopsis: This module contains a class used to plot the results
              related to the third experiment of the thesis conducted
              on the bug reports of Mozilla Firefox. The experiment
              consists mainly of comparing several feature selection
              techniques and selecting the best one.

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
from experiment_3.experiment_3_results_plotter \
import Exp3ResultsPlotter

class MozillaExp3ResultsPlotter(Exp3ResultsPlotter):

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
                "x_lim_min": 0.136,
                "x_lim_max": 0.1735,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "title_font_size": 60,
                "labels_font_size": 50,
                "y_tick_labels_font_size": 40,
                "title": "Accuracy of the different feature " + \
                "selection techniques",
                "file_name" : "experiment_31.png",
                "debug_title": "Average Accuracy",
                "bars_labels_space": 0.0004
            },
            {
                "key": "avg_mrr",
                "x_lim_min": 0.312,
                "x_lim_max": 0.3525,
                "x_label": "MRR",
                "y_label": "Configurations",
                "title_font_size": 60,
                "labels_font_size": 50,
                "y_tick_labels_font_size": 40,
                "title": "MRR of the different feature selection " + \
                "techniques",
                "file_name" : "experiment_32.png",
                "debug_title": "Average MRR",
                "bars_labels_space": 0.0004
            }
        ]
        super().plot_results()

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_feature_selection_" + \
    "experiment_results.json"
    mozilla_exp_3_results_plotter = \
    MozillaExp3ResultsPlotter(cleaned_results_file_name)
    mozilla_exp_3_results_plotter.plot_results()

if __name__ == "__main__":
    main()
