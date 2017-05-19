# -*- coding: utf-8 -*-
"""
.. module:: mozilla_data_set_sorter
   :platform: Unix, Windows
   :synopsis: This module is used to sort the Mozilla Firefox data set
   used in this thesis (the data should have been scrapped via the
   Scrapy library and in a JSON file).

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)
os.sys.path.insert(0, grand_parent_dir)
from scrap.data_set_sorter import DataSetSorter

def main():
	current_dir = os.path.dirname(os.path.abspath( \
    inspect.getfile(inspect.currentframe())))
	os.path.dirname(current_dir)
	data_set_sorter = DataSetSorter("brs.json", "sorted_brs.json")
	data_set_sorter.write_sorted_data()

if __name__ == "__main__":
    main()