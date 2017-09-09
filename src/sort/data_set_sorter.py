# -*- coding: utf-8 -*-
"""
.. module:: data_set_sorter
   :platform: Unix, Windows
   :synopsis: This module contains a class used to sort any data set 
   used in this thesis (the data should have been scrapped via the 
   Scrapy library and in a JSON file).

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import json

class DataSetSorter:

    def __init__(self, data_file_path, output_file_path):
        """Constructor"""
        self.content = None
        self.data_file_path = data_file_path
        self.output_file_path = output_file_path

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, new_content):
        self._content = new_content

    @property
    def data_file_path(self):
        return self._data_file_path

    @data_file_path.setter
    def data_file_path(self, new_data_file_path):
        self._data_file_path = new_data_file_path

    @property
    def output_file_path(self):
        return self._output_file_path

    @output_file_path.setter
    def output_file_path(self, new_output_file_path):
        self._output_file_path = new_output_file_path

    def write_sorted_data(self):
        """Sorts the data set and dumps it"""
        self._load_json_file()
        self._sort_data_set()
        self._write_data()

    def _load_json_file(self):
        """Opens and loads the data from a JSON file."""
        print("Loading...") # Debug
        with open(self.data_file_path) as json_data:
            self._content = json.load(json_data)
        print("Loaded") # Debug

    def _sort_data_set(self):
        """Sorts the "content" list"""
        print("Number of BRs before sorting: {}" \
              .format(len(self._content))) # Debug
        print("Sorting...") # Debug
        self._content.sort(key=lambda br: int(br["bug_id"][4:]))
        print("Sorted") # Debug
        print("Number of BRs after sorting: {}" \
              .format(len(self._content))) # Debug

    def _write_data(self):
        """Writes the data which are in the "content" attribute 

        It writes the data into a JSON file which path is in the
        "output_file" attribute of our object.
        """
        print("Writing...") # Debug
        with open(self.output_file_path, 'w') as output_file:
            json.dump(self._content, output_file, indent=4)
        print("Written") # Debug

def main():
    data_set_sorter = DataSetSorter("brs.json", "sorted_brs.json")
    data_set_sorter.write_sorted_data()

if __name__ == "__main__":
    main()