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

def load_cleaned_results(cleaned_results_file_name):
    """Opens and loads the cleaned results from a JSON file."""
    return load_json_file(cleaned_results_file_name)

def main():
    """The main function of the script"""
    
    cleaned_results_file_name = "cleaned_tr_" + \
    "representation_experiment_results.json"
    
    cleaned_results = load_cleaned_results(cleaned_results_file_name)
        
    # The dictionary below contains the accuracies related to the 
    # different representations (only when not combining features)
    accuracies_1 = cleaned_results["not_combined_avg_accuracy"]

    # The dictionary below contains the accuracies related to the 
    # different representations (only when combining features)
    accuracies_2 = cleaned_results["combined_avg_accuracy"]

    # Below, we generate two lists of tuples based on the above
    # dictionaries. The first element of each tuple is the reduced
    # form of the original key. The second element is the accuracy.
    list_of_accuracies_1 = \
    get_cleaned_list_from_dict_without_combination(accuracies_1)
    list_of_accuracies_2 = \
    get_cleaned_list_from_dict_with_combination(accuracies_2)

    # We sort the aforementioned cleaned lists in the reverse order
    list_of_accuracies_1.sort(key=lambda x: x[1], reverse=True)
    list_of_accuracies_2.sort(key=lambda x: x[1], reverse=True)

    # Debug
    for acronym, accuracy in list_of_accuracies_1:
        print("{}: {}".format(acronym, accuracy))

    # Debug
    for acronym, accuracy in list_of_accuracies_2:
        print("{}: {}".format(acronym, accuracy))

    # Below, we generate two lists from the first above mentioned list
    # of tuples
    y_labels, x_values = generate_two_lists(list_of_accuracies_1)

    # Below, we make a bar plot and save it into a file
    file_name = "experiment_21.png"
    title = "Accuracy of the different feature extraction "+ \
    "techniques (without combination of features)"
    plot_bar(title, y_labels, x_values, 0.32, 0.79, file_name)

    # Below, we generate two lists from the second above mentioned
    # list of tuples
    y_labels, x_values = generate_two_lists(list_of_accuracies_2)

    # Below, we make a bar plot and save it into a file
    file_name = "experiment_22.png"
    title = "Accuracy of the different feature extraction " + \
    "techniques (with combination of features)"
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

def get_cleaned_list_from_dict_without_combination(dict_content):
    """Returns a cleaned list from a dictionary

    The aforementioned dictionary should only contain the accuracies
    related to different feature extraction techniques (without
    combination)
    """
    generated_list = []
    for key, value in dict_content.items():
        if isinstance(value, list):
            i = 0
            for accuracy in value:
                generated_list.append((get_small_key(key) + \
                    "-{}".format(10+i*20), accuracy))
                i += 1
        else:
            generated_list.append((get_small_key(key), value))        
    return generated_list

def get_cleaned_list_from_dict_with_combination(dict_content):
    """Returns a cleaned list from a dictionary

    The aforementioned dictionary should only contain the accuracies
    related to different feature extraction techniques (with
    combination)
    """
    generated_list = []
    for key, value in dict_content.items():
        key_split = key.split("_")
        generated_list.append((get_small_key(key_split[0]) + "&" + get_small_key(key_split[1]), value))
    return generated_list

def get_small_key(key):
    """Return the reduced form of a key"""
    return KEY_MAPPING[key]

if __name__ == "__main__":
    main()