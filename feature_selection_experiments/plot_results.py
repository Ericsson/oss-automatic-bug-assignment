# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from utilities import load_json_file

# Below, there is a dictionary which maps each key to a smaller one 
# for readability purposes
KEY_MAPPING = {
    "GridSearch chi2 Linear SVM": "CHI-2",
    "GridSearch f_classif Linear SVM": "ANOVA",
    "GridSearch mutual_info_classif Linear SVM": "MI",
    "RFECV SVM": "RFE"
}

def load_cleaned_results(cleaned_results_file_name):
    """Opens and loads the cleaned results from a JSON file."""
    return load_json_file(cleaned_results_file_name)

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_feature_" + \
    "selection_experiment_results.json"
    
    cleaned_results = load_cleaned_results(cleaned_results_file_name)
        
    # The dictionary below contains the accuracies related to the 
    # different configurations
    accuracies = cleaned_results["avg_accuracy"]

    # Below, we generate a list of tuples based on the above
    # dictionary. The first element of each tuple is the reduced form
    # of the original key. The second element is the accuracy.
    list_of_accuracies = get_cleaned_list_from_dict(accuracies)

    # We sort the aforementioned cleaned list in the reverse order
    list_of_accuracies.sort(key=lambda x: x[1], reverse=True)

    # Debug
    for acronym, accuracy in list_of_accuracies:
        print("{}: {}".format(acronym, accuracy))

    # Below, we generate two lists from the above mentioned list of
    # tuples
    y_labels, x_values = generate_two_lists(list_of_accuracies)

    file_name = "experiment_3.png"
    plot_bar(y_labels, x_values, 0.74, 0.756, file_name)

def plot_bar(y_labels, x_values, x_lim_min=0.32, x_lim_max=0.79, file_name=None):
    """Makes a bar plot"""
    # We get a subplot inside a figure and its associated axis
    fig, ax = plt.subplots()

    # We set the size of the figure
    fig.set_size_inches(25, 40)

    height = 0.9 # Height of each bin

    # Below, we compute the position of each bin
    bins = list(map(lambda y: y, range(1, len(y_labels)+1)))

    plt.title("Accuracy of the different feature selection " + \
    "techniques", fontsize=40)

    # We set the x and y limits of the chart
    plt.ylim([1-height, len(y_labels) + height])
    plt.xlim([x_lim_min, x_lim_max])

    # We set the labels of the chart
    plt.ylabel("Configurations", fontsize=35)
    plt.xlabel("Accuracy", fontsize=35)
    
    # We make the bar plot
    ax.barh(bins, x_values, height=height, alpha=0.8)
    
    # We set the x-ticks and the labels of the x-ticks 
    ax.set_yticks(list(map(lambda y: y, range(1,len(y_labels)+1))))
    ax.set_yticklabels(y_labels, fontsize=30)

    add_labels_to_bars(x_values, ax)

    save_file = None if not file_name \
    else os.path.join(current_dir, file_name)  

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def generate_two_lists(list_of_two_elements_tuples):
    """Returns two lists from a list of tuples"""
    list_1 = []
    list_2 = []
    for element_1, element_2 in list_of_two_elements_tuples:
        list_1.append(element_1)
        list_2.append(element_2)
    return list_1, list_2

def add_labels_to_bars(x_values, ax):
    """Add labels (percentage) to some given bars"""
    for i, v in enumerate(x_values):
        ax.text(v + 0.0002, i+1, "{:.2%}".format(v), color='k', \
            fontsize=25, verticalalignment='center')

def get_cleaned_list_from_dict(dict_content):
    """Returns a cleaned list from a dictionary

    The aforementioned dictionary should only contain the accuracies
    related to different feature selection techniques"""
    generated_list = []
    for key, value in dict_content.items():
        if key == "RFECV SVM":
            i = 0
            for i in range(5):
                generated_list.append((get_small_key(key) + \
                    "-{}".format(10+i*20), value[2+i*2]))
        else:
            i = 0
            for accuracy in value:
                generated_list.append((get_small_key(key) + \
                    "-{}".format(10+i*20), accuracy))
                i += 1
    return generated_list

def get_small_key(key):
    """Return the reduced form of a key"""
    return KEY_MAPPING[key]

if __name__ == "__main__":
    main()