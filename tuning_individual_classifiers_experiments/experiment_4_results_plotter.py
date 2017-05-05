import os
import inspect
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir) 
from results_plotter import ResultsPlotter

class Experiment4ResultsPlotter(ResultsPlotter):

    # Below, there is a dictionary which maps each key to a smaller 
    # one for readability purposes
    KEY_MAPPING = {
        "NearestCentroid": "NC",
        "MultinomialNB": "MNB",
        "LinearSVC": "LSVC",
        "Primal LogisticRegression": "PLR",
        "Dual LogisticRegression": "DLR",
        "PerceptronWithPenalty": "PWP",
        "PerceptronWithoutPenalty": "P",
        "SGDClassifier": "SGCD"
    }
    
    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        super().__init__(cleaned_results_file_name)
    
    @classmethod
    def get_cleaned_list_from_dict(cls, dict_content):
        """Returns two cleaned lists from a dictionary
    
        The aforementioned dictionary should contain some sub-dictionaries
        containing the parameters, the MRR values and the accuracies 
        related to the different configurations.
        """
        generated_accuracy_list = []
        generated_mrr_list = []
        # Below, we iterate over each ML algorithm
        for key, value in dict_content.items():
            means_accuracy = value["means_accuracy"]
            means_mrr = value["means_mrr"]
            params_list = value["params"]
            # Below, we iterate over each ML algorithm's configuration
            for i, params in enumerate(params_list):
                accuracy = means_accuracy[i]
                mrr = means_mrr[i]
                cleaned_key = cls.get_small_key(key)
                temp = cleaned_key
                # Below, we iterate to build a unique key based on the 
                # parameters of the current configuration
                for param_key, param_value in params.items():
                    param_key_list = param_key.split("__")
                    cleaned_key += ("_" + param_key_list[1] + "=" + \
                                    str(param_value)) 
                generated_accuracy_list.append((cleaned_key, accuracy))
                if temp != "NC":
                    generated_mrr_list.append((cleaned_key, mrr))
        return generated_accuracy_list, generated_mrr_list
    
    @classmethod
    def get_small_key(cls, key):
        """Return the reduced form of a key"""
        return cls.KEY_MAPPING[key]

    def plot_results(self):
        """This method plots the results in chart(s)"""
        super().plot_results()
        plot_parameters = [
            {
                "key": "normal_avg",
                "x_lim_min": [0, 0],
                "x_lim_max": [1, 1],
                "x_label": ["Accuracy", "MRR"],
                "y_label": ["Configurations", "Configurations"],
                "labels_font_size": [35, 35], 
                "y_tick_labels_font_size": [20, 20],
                "bars_labels_space": [0.0005, 0.0005], 
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
                "x_lim_min": [0, 0],
                "x_lim_max": [1, 1],
                "x_label": ["Accuracy", "MRR"],
                "y_label": ["Configurations", "Configurations"],
                "labels_font_size": [35, 35], 
                "y_tick_labels_font_size": [20, 20],
                "bars_labels_space": [0.0005, 0.0005],
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
        
        for plot_parameter in plot_parameters:
            # The dictionary below contains the values of two metric 
            # related to the different configurations
            dict_of_metric_values = self. \
            _cleaned_results[plot_parameter["key"]]
            
            # Below, we generate two lists of tuples based on the 
            # above dictionary. The first element of each tuple is the 
            # reduced form of the original key. The second element is
            # the metric values.
            list_of_accuracy_values, list_of_mrr_values = self. \
            get_cleaned_list_from_dict(dict_of_metric_values)
            
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_accuracy_values.sort(key=lambda x: x[1], \
                                       reverse=True)
            
            self.print_dict(plot_parameter["debug_title"][0], \
                            list_of_accuracy_values) # Debug
            
            # Below, we generate a list from the first above mentioned 
            # lists of tuples
            y_labels, x_values = self. \
            generate_two_lists(list_of_accuracy_values)
            
            x_lim_min = plot_parameter["x_lim_min"][0]
            x_lim_max = plot_parameter["x_lim_max"][0]
            x_label = plot_parameter["x_label"][0]
            y_label = plot_parameter["y_label"][0]
            labels_font_size = plot_parameter["labels_font_size"][0]
            y_tick_labels_font_size = \
            plot_parameter["y_tick_labels_font_size"][0]
            title = plot_parameter["title"][0]
            file_name = plot_parameter["file_name"][0]
            bars_labels_space = plot_parameter["bars_labels_space"][0]
            self.plot_bar(y_labels, x_values, x_lim_min, x_lim_max, \
                          x_label, y_label, title, file_name, \
                          labels_font_size, y_tick_labels_font_size, \
                          bars_labels_space)
            
            self.print_dict(plot_parameter["debug_title"][1], \
                            list_of_mrr_values) # Debug
            
            # Below, we generate a list from the second above 
            # mentioned lists of tuples
            y_labels, x_values = self. \
            generate_two_lists(list_of_mrr_values)
            
            x_lim_min = plot_parameter["x_lim_min"][1]
            x_lim_max = plot_parameter["x_lim_max"][1]
            x_label = plot_parameter["x_label"][1]
            y_label = plot_parameter["y_label"][1]
            labels_font_size = plot_parameter["labels_font_size"][1]
            y_tick_labels_font_size = \
            plot_parameter["y_tick_labels_font_size"][1]
            title = plot_parameter["title"][1]
            file_name = plot_parameter["file_name"][1]
            bars_labels_space = plot_parameter["bars_labels_space"][1]
            self.plot_bar(y_labels, x_values, x_lim_min, x_lim_max, \
                          x_label, y_label, title, file_name, \
                          labels_font_size, y_tick_labels_font_size, \
                          bars_labels_space)