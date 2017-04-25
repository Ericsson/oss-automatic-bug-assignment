import matplotlib.pyplot as plt
import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir) 
from utilities import load_json_file

# Below, there is a list of tuples to map the names of the different
# pre-processing steps to their associated acronyms
ACRONYMS_MAPPING = [ \
    (1, "C"), \
    (3, "S"), \
    (5, "L"), \
    (7, "SW"), \
    (11, "P"), \
    (14, "N"), \
    (17, "LC")
]

def load_cleaned_results(cleaned_results_file_name):
    """Opens and loads the cleaned results from a JSON file."""
    return load_json_file(cleaned_results_file_name)

def main():
    """The main function of the script"""
    cleaned_results_file_name = "cleaned_pre_processing_" + \
    "experiment_results.json"
    
    cleaned_results = load_cleaned_results(cleaned_results_file_name)
    
    # Below, we generate a list of tuples based on the above
    # dictionary. The first element of each tuple is the reduced form
    # of the associated file name. The second element is the accuracy.
    list_of_accuracies = get_cleaned_list_from_dict(cleaned_results["avg_accuracy"])

    # We sort the aforementioned cleaned list in the reverse order
    list_of_accuracies.sort(key=lambda x: x[1], reverse=True)

    # Debug
    for acronym, accuracy in list_of_accuracies:
        print("{}: {}".format(acronym, accuracy))

    # Below, we generated two lists from above mentioned list of tuples 
    y_labels, x_values = generate_two_lists(list_of_accuracies)

    file_name = "experiment_1.png"
    plot_bar(y_labels, x_values, file_name)

def plot_bar(y_labels, x_values, file_name=None):
    """Makes a bar plot"""
    # We get a subplot inside a figure and its associated axis
    fig, ax = plt.subplots()

    # We set the size of the figure
    fig.set_size_inches(25, 40)

    height = 0.9 # Height of each bin

    # Below, we compute the position of each bin
    bins = list(map(lambda y: y, range(1, len(y_labels)+1)))

    plt.title("Accuracy of the different pre-processing " + \
        "configurations", fontsize=40)

    # We set the x and y limits of the chart
    plt.ylim([1-height, len(y_labels) + height])
    plt.xlim([0.59, 0.7560])

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
        ax.text(v + 0.0005, i+1, "{:.2%}".format(v), color='k', fontsize=25, verticalalignment='center')

def get_cleaned_list_from_dict(dict_content):
    """Returns a cleaned list from a dictionary"""
    generated_list = []
    for key, value in dict_content.items():
        generated_list.append((get_small_name(key), value))
    return generated_list

def get_small_name(name_of_file):
    """Return the reduced form of a file name"""
    name_of_file_parts = name_of_file.split(".")
    name_of_file_without_extension = name_of_file_parts[0]
    name_of_file_parts = name_of_file_without_extension.split("_")
    small_name = ""
    # print(name_of_file) # Debug
    for index, acronym in ACRONYMS_MAPPING:
        # print(acronym) # Debug
        # print(name_of_file_parts[index]) # Debug        
        if index > 1:
            small_name += "|"
        small_name += acronym if name_of_file_parts[index] ==  "with" else "NOT({})".format(acronym)
    return small_name

if __name__ == "__main__":
    main()