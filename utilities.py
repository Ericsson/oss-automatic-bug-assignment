# -*- coding: utf-8 -*-

from pandas import DataFrame
import json
import pandas as pd
import logging

def print_log(string):
    print(string)
    logging.info(string)

def load_json_file(json_file_path):
    """Opens and loads the data from a JSON file."""
    with open(json_file_path) as json_data:
        return json.load(json_data)

def load_data_set(data_set_file_path):
    """Opens and loads the data set from a JSON file."""
    return load_json_file(data_set_file_path)
        
def load_developers_mappings(developers_mappings_file_path):
    """Opens and loads the developers mappings from a JSON file."""
    return load_json_file(developers_mappings_file_path)
        
def load_distinct_developers_list(developers_list_file_path):
    """Opens and loads the distinct developers from a JSON file."""
    return load_json_file(developers_list_file_path)
    
def build_data_frame(loaded_data, developers_dict_data, developers_list_data):
    """Build a data frame.
    
    It uses the loaded data set, the loaded developers mappings, the loaded 
    distinct developers.
    """
    rows = []
    index = []
    # Then, we build a data frame 
    print("Loading the pre-processed data") # Debug
    for element in loaded_data:
        if len(element['tr_id']) != 0:
            rows.append({
                "text": " ".join(element['heading']) + \
                " " + " ".join(element['observation']), 
                "class": element['devloper']
                })
            # print(element['tr_id']) # Debug
            index.append(element['tr_id'])
    # We build a Data Frame with the pre-processed data
    data_frame = DataFrame(rows, index=index)
    print("Shape of the Data Frame") # Debug
    print(data_frame.shape) # Debug
    print("Data Frame is being filtered") # Debug
    # We only keep the BRs which developers are in the loaded list
    df1 = data_frame[data_frame['class'] \
    .isin(developers_list_data)]
    print("Shape of the Data Frame 1") # Debug
    print(df1.shape)
    df2 = data_frame[data_frame['class'] \
    .str.contains(r"X.*")]
    print("Shape of the Data Frame 2") # Debug
    print(df2.shape)
    df3 = data_frame[data_frame['class'] \
    .str.contains(r"Y.*")]
    print("Shape of the Data Frame 3") # Debug
    print(df3.shape)

    df = pd.concat([df1, df2, df3])
    df = df[~df.index \
    .duplicated(keep="first")]

    print("Shape of the filtered Data Frame") # Debug
    print(df.shape) # Debug
    print("The developers are mapped") # Debug
    df["class"] \
    .replace(developers_dict_data, inplace=True, regex=True)
    print("Shape of the mapped Data Frame") # Debug
    print(df.shape) # Debug
    print("Data Frame first lines") # Debug
    print(df.head()) # Debug
    # print("We shuffle the rows of the Data Frame") # Debug
    # df = df \
    # .reindex(np.random.permutation(df.index))
    print("Data Frame first lines") # Debug
    print(df.head()) # Debug
    print("Number of classes in the Data Frame") # Debug
    print(len(set(df['class'].values))) # Debug
    print("The classes in the Data Frame") # Debug
    print(set(df['class'].values)) # Debug

    # Some tests
    print("Debugging")
    print("W" in set(df['class'].values))
    print("X" in set(df['class'].values))
    print("Y" in set(df['class'].values))
    print("Z" in set(df['class'].values))
    print("W" in set(df['class'].values))
    print("Frequency of each class")
    print(pd.value_counts(df['class'].values))
    
    return df
