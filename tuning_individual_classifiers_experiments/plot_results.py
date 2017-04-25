import matplotlib.pyplot as plt
import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from utilities import load_json_file

# Below, there is a dictionary which map each key to a smaller one 
# for readability purposes
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

def load_cleaned_results(cleaned_results_file_name):
    """Opens and loads the cleaned results from a JSON file."""
    return load_json_file(cleaned_results_file_name)

def main():
    """The main function of the script"""
    
    cleaned_results_file_name = "cleaned_tuning_individual_" + \
    "classifier_generic_experiment_results.json"
    
    cleaned_results = load_cleaned_results(cleaned_results_file_name)
        
    # The dictionary below contains the accuracies related to the 
    # different configurations (grid search)
    normal_accuracies = cleaned_results["normal_avg_accuracy"]

    # The dictionary below contains the accuracies related to the 
    # different configurations (random search)
    random_accuracies = cleaned_results["random_avg_accuracy"]

    # Below, we generate two lists of tuples based on the above
    # dictionaries. The first element of each tuple is the reduced
    # form of the original key. The second element is the accuracy
    list_of_normal_accuracies = \
    get_cleaned_list_from_dict(normal_accuracies)
    list_of_random_accuracies = \
    get_cleaned_list_from_dict(random_accuracies)

    # We sort the aforementioned cleaned lists in the reverse order
    list_of_normal_accuracies.sort(key=lambda x: x[1], reverse=True)
    list_of_random_accuracies.sort(key=lambda x: x[1], reverse=True)

    # Debug
    for acronym, accuracy in list_of_normal_accuracies:
        print("{}: {}".format(acronym, accuracy))

    # Debug
    for acronym, accuracy in list_of_random_accuracies:
        print("{}: {}".format(acronym, accuracy))

    # Below, we generate two lists from the first above mentioned list
    # of tuples
    y_labels, x_values = generate_two_lists(list_of_normal_accuracies)

    # Below, we make a bar plot and save it into a file
    file_name = "experiment_41.png"
    title = "Accuracy of the different configurations "+ \
    "(grid search)"
    plot_bar(title, y_labels, x_values, 0.32, 0.79, file_name)

    # Below, we generate two lists from the second above mentioned
    # list of tuples
    y_labels, x_values = generate_two_lists(list_of_random_accuracies)

    # Below, we make a bar plot and save it into a file
    file_name = "experiment_42.png"
    title = "Accuracy of the different configurations " + \
    "(random search)"
    plot_bar(title, y_labels, x_values, 0.55, 0.79, file_name)

def plot_bar(title, y_labels, x_values, x_lim_min=0.32, x_lim_max=0.79, file_name=None):
    """Makes a bar plot"""
    # We get a subplot inside a figure and its associated axis
    fig, ax = plt.subplots()

    # We set the size of the figure
    fig.set_size_inches(25, 40)

    height = 0.9 # Height of each bin

    # Below, we compute the position of each bin
    bins = list(map(lambda y: y, range(1, len(y_labels)+1)))

    plt.title(title, fontsize=40)

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

    # Debug
    print(file_name)

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
        ax.text(v + 0.0005, i+1, "{:.2%}".format(v), color='k', \
            fontsize=25, verticalalignment='center')

def get_cleaned_list_from_dict(dict_content):
    """Returns a cleaned list from a dictionary

    The aforementioned dictionary should contain some sub-dictionaries
    containing the parameters and the accuracies related to the
    different configurations.
    """
    generated_list = []
    # Below, we iterate over each ML algorithm
    for key, value in dict_content.items():
        means = value["means"]
        params_list = value["params"]
        # Below, we iterate over each ML algorithm's configuration
        for i, params in enumerate(params_list):
            accuracy = means[i]
            cleaned_key = get_small_key(key)
            # Below, we iterate to build a unique key based on the 
            # parameters of the current configuration
            for param_key, param_value in params.items():
                param_key_list = param_key.split("__")
                cleaned_key += ("_" + param_key_list[1] + "=" + \
                                str(param_value)) 
            generated_list.append((cleaned_key, accuracy))
    return generated_list

def get_small_key(key):
    """Return the reduced form of a key"""
    return KEY_MAPPING[key]

if __name__ == "__main__":
    main()