import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from feature_selection_experiments.experiment_3_results_plotter \
import Experiment3ResultsPlotter

class MozillaExperiment3ResultsPlotter(Experiment3ResultsPlotter):
    
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
                "x_lim_min": 0.135,
                "x_lim_max": 0.1725,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 30,
                "title": "Accuracy of the different feature " + \
                "selection techniques",
                "file_name" : "experiment_31.png",
                "debug_title": "Average Accuracy",
                "bars_labels_space": 0.0002
        
            },
            {
                "key": "avg_mrr",
                "x_lim_min": 0.31,
                "x_lim_max": 0.3525,
                "x_label": "MRR",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 30,
                "title": "MRR of the different feature selection " + \
                "techniques",
                "file_name" : "experiment_32.png",
                "debug_title": "Average MRR",
                "bars_labels_space": 0.0002
            }
        ]
        super().plot_results()
       
def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_feature_selection_" + \
    "experiment_results.json"
    experiment_3_results_plotter = \
    MozillaExperiment3ResultsPlotter(cleaned_results_file_name)
    experiment_3_results_plotter.plot_results()

if __name__ == "__main__":
    main()
