import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from tuning_individual_classifiers_experiments.experiment_4_results_plotter \
import Experiment4ResultsPlotter

class EclipseExperiment4ResultsPlotter(Experiment4ResultsPlotter):
    
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        self._current_dir = os.path.dirname(os.path.abspath( \
        inspect.getfile(inspect.currentframe())))
        super().__init__(cleaned_results_file_name)

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_tuning_individual_" + \
    "classifier_generic_experiment_results.json"
    experiment_4_results_plotter = \
    EclipseExperiment4ResultsPlotter(cleaned_results_file_name)
    experiment_4_results_plotter.plot_results()

if __name__ == "__main__":
    main()