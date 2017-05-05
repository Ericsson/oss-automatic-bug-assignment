import os
import inspect
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from results_plotter import ResultsPlotter

class Experiment2ResultsPlotter(ResultsPlotter):

    # Below, there is a dictionary which map each key to a smaller one 
    # for readability purposes
    KEY_MAPPING = {
        "Boolean SVM": "BOOL",
        "TF SVM": "TF",
        "TF IDF SVM": "TF-IDF",
        "GridSearch Boolean Truncated SVD SVM": "BOOL+SVD",
        "GridSearch TF Truncated SVD SVM": "TF+SVD",
        "GridSearch TF IDF Truncated SVD SVM": "TF-IDF+SVD",
        "GridSearch Boolean NMF SVM": "BOOL+NMF",
        "GridSearch TF NMF SVM": "TF+NMF",
        "GridSearch TF IDF NMF SVM": "TF-IDF+NMF",
    }
    
    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        super().__init__(cleaned_results_file_name)
    
    @classmethod
    def get_cleaned_list_from_dict_without_combination(cls, dict_content):
        """Returns a cleaned list from a dictionary
    
        The aforementioned dictionary should only contain the metric 
        values related to different feature extraction techniques 
        (without combination)
        """
        generated_list = []
        for key, value in dict_content.items():
            if isinstance(value, list):
                i = 0
                for accuracy in value:
                    generated_list.append((cls.get_small_key(key) + \
                        "-{}".format(10+i*20), accuracy))
                    i += 1
            else:
                generated_list.append((cls.get_small_key(key), value))        
        return generated_list
    
    @classmethod
    def get_cleaned_list_from_dict_with_combination(cls, \
                                                    dict_content):
        """Returns a cleaned list from a dictionary
    
        The aforementioned dictionary should only contain the metric 
        values related to different feature extraction techniques 
        (with combination)
        """
        generated_list = []
        for key, value in dict_content.items():
            key_split = key.split("_")
            generated_list.append((cls.get_small_key(key_split[0]) + \
                                   "&" + \
                                   cls.get_small_key(key_split[1]), \
                                   value))
        return generated_list
    
    @classmethod
    def get_small_key(cls, key):
        """Return the reduced form of a key"""
        return cls.KEY_MAPPING[key]

    def plot_results(self):
        """This method plots the results in chart(s)"""
        super().plot_results()
        plot_parameters = [
            {
                "key": "not_combined_avg_accuracy",
                "x_lim_min": 0.34,
                "x_lim_max": 0.79,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 30,
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
                "x_lim_min": 0.56,
                "x_lim_max": 0.78,
                "x_label": "Accuracy",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 30,
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
                "x_lim_min": 0.52,
                "x_lim_max": 0.8675,
                "x_label": "MRR",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 30,
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
                "x_lim_min": 0.70,
                "x_lim_max": 0.86,
                "x_label": "MRR",
                "y_label": "Configurations",
                "labels_font_size": 35, 
                "y_tick_labels_font_size": 30,
                "title": "MRR of the different feature " + \
                "extraction techniques (with combination of " + \
                "features)",
                "file_name": "experiment_24.png",
                "debug_title": "Average MRR (With Combination)",
                "combined": True,
                "bars_labels_space": 0.0005
            }   
        ]
        
        for plot_parameter in plot_parameters:
            # The dictionary below contains the values of a metric 
            # related to the different representations
            dict_of_metric_values = self. \
            _cleaned_results[plot_parameter["key"]]
            
            # Below, we generate a list of tuples based on the above 
            # dictionary. The first element of each tuple is the 
            # reduced form of the original key. The second element is 
            # the metric value.
            if plot_parameter["combined"]:
                list_of_metric_values = \
                self.get_cleaned_list_from_dict_with_combination( \
                dict_of_metric_values)

            else:
                list_of_metric_values = \
                self.get_cleaned_list_from_dict_without_combination( \
                dict_of_metric_values)
    
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_metric_values.sort(key=lambda x: x[1], \
                                       reverse=True)
    
            self.print_dict(plot_parameter["debug_title"], \
                            list_of_metric_values) # Debug

    
            # Below, we generate two lists from the first above mentioned list
            # of tuples
            y_labels, x_values = self. \
            generate_two_lists(list_of_metric_values)
            
            x_lim_min = plot_parameter["x_lim_min"]
            x_lim_max = plot_parameter["x_lim_max"]
            x_label = plot_parameter["x_label"]
            y_label = plot_parameter["y_label"]
            labels_font_size = plot_parameter["labels_font_size"]
            y_tick_labels_font_size = \
            plot_parameter["y_tick_labels_font_size"]
            title = plot_parameter["title"]
            file_name = plot_parameter["file_name"]
            bars_labels_space = plot_parameter["bars_labels_space"]
            self.plot_bar(y_labels, x_values, x_lim_min, x_lim_max, \
                          x_label, y_label, title, file_name, \
                          labels_font_size, y_tick_labels_font_size, \
                          bars_labels_space)