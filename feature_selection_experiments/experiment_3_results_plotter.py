import os
import inspect
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir) 
from results_plotter import ResultsPlotter

class Experiment3ResultsPlotter(ResultsPlotter):
    
    # Below, there is a dictionary which maps each key to a smaller 
    # one for readability purposes
    KEY_MAPPING = {
        "GridSearch chi2 Linear SVM": "CHI-2",
        "GridSearch f_classif Linear SVM": "ANOVA",
        "GridSearch mutual_info_classif Linear SVM": "MI",
        "RFECV SVM": "RFE"
    }

    @abc.abstractmethod
    def __init__(self, cleaned_results_file_name):
        """Constructor"""
        super().__init__(cleaned_results_file_name)
        
    @classmethod
    def get_cleaned_list_from_dict(cls, dict_content):
        """Returns a cleaned list from a dictionary
    
        The aforementioned dictionary should only contain the 
        accuracies related to different feature selection 
        techniques"""
        generated_list = []
        for key, value in dict_content.items():
            if key == "GridSearch var_threshold Linear SVM":
                continue
            if key == "RFECV SVM":
                i = 0
                for i in range(5):
                    generated_list.append((cls.get_small_key(key) + \
                        "-{}".format(10+i*20), value[2+i*2]))
            else:
                i = 0
                for accuracy in value:
                    generated_list.append((cls.get_small_key(key) + \
                        "-{}".format(10+i*20), accuracy))
                    i += 1
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
                "key": "avg_accuracy",
                "x_lim_min": 0.7435,
                "x_lim_max": 0.758,
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
                "x_lim_min": 0.832,
                "x_lim_max": 0.842,
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
        
        for plot_parameter in plot_parameters:
            # The dictionary below contains the values of a metric 
            # related to the different configurations
            dict_of_metric_values = self. \
            _cleaned_results[plot_parameter["key"]]
            
            # Below, we generate a list of tuples based on the above
            # dictionary. The first element of each tuple is the 
            # reduced form of the original key. The second element is
            # the metric values.
            list_of_metric_values = self. \
            get_cleaned_list_from_dict(dict_of_metric_values)
            
            # We sort the aforementioned cleaned list in the reverse 
            # order
            list_of_metric_values.sort(key=lambda x: x[1], \
                                       reverse=True)
            
            self.print_dict(plot_parameter["debug_title"], \
                            list_of_metric_values) # Debug
            
            # Below, we generate two lists from the above mentioned 
            # list of tuples
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