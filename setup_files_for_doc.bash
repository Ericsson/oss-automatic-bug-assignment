#!/bin/bash

echo "Hello World!"

sphinx-quickstart

read -n1 -rsp "Please modify the conf.py file before continuing"
echo ""
echo "Thanks"

echo "Now, please wait..."
sphinx-apidoc -f -o ./doc/source ./src ./doc
printf "..."
sphinx-apidoc -f -d 10 --implicit-namespaces -o ./doc/source ./src ./doc

printf "scikit\_learn namespace\n"\
"=======================\n"\
"\n"\
"Submodules\n"\
"----------\n"\
"\n"\
"scikit\_learn\.accuracy\_mrr\_scoring\_object module\n"\
"----------------------------------------------------\n"\
"\n"\
".. automodule:: scikit_learn.accuracy_mrr_scoring_object\n"\
"    :members:\n"\
"    :undoc-members:\n"\
"    :show-inheritance:\n"\
"\n"\
"scikit\_learn\.rfe module\n"\
"-------------------------\n"\
"\n"\
".. automodule:: scikit_learn.rfe\n"\
"    :undoc-members:\n"\
"    :show-inheritance:\n\n" > ./doc/source/scikit_learn.rst

echo "..."

printf "oss_automatic_bug_assignment\n"\
"============================\n"\
"\n"\
"Subpackages\n"\
"-----------\n"\
"\n"\
".. toctree::\n"\
"   :maxdepth: 10\n"\
"  \n"\
"   eclipse_jdt\n"\
"   feature_selection_experiments\n"\
"   mozilla_firefox\n"\
"   pre_processing_experiments\n"\
"   scikit_learn\n"\
"   scrap\n"\
"   size_of_data_set_experiments\n"\
"   tr_representation_experiments\n"\
"   tuning_individual_classifiers_experiments\n"\
"\n"\
"Submodules\n"\
"----------\n"\
"\n"\
".. toctree::\n"\
"\n"\
"   classify\n"\
"   classify_k_folds\n"\
"   experiment\n"\
"   experiments_launcher\n"\
"   results_plotter\n"\
"   utilities\n\n" > ./doc/source/modules.rst

echo "..."

printf "eclipse\_jdt namespace\n"\
"======================\n"\
"\n"\
"Subpackages\n"\
"-----------\n"\
"\n"\
".. toctree::\n"\
"\n"\
"    eclipse_jdt.feature_selection_experiments\n"\
"    eclipse_jdt.pre_processing_experiments\n"\
"    eclipse_jdt.scrap_eclipse_jdt\n"\
"    eclipse_jdt.size_of_data_set_experiments\n"\
"    eclipse_jdt.tr_representation_experiments\n"\
"    eclipse_jdt.tuning_individual_classifiers_experiments\n"\
"\n"\
"Submodules\n"\
"----------\n"\
"\n"\
"eclipse\_jdt\.eclipse\_experiments\_launcher module\n"\
"---------------------------------------------------\n"\
"\n"\
".. automodule:: eclipse_jdt.eclipse_experiments_launcher\n"\
"    :members:\n"\
"    :undoc-members:\n"\
"    :show-inheritance:\n\n" > ./doc/source/eclipse_jdt.rst

echo "..."

printf "mozilla\_firefox namespace\n"\
"==========================\n"\
"\n"\
"Subpackages\n"\
"-----------\n"\
"\n"\
".. toctree::\n"\
"\n"\
"    mozilla_firefox.feature_selection_experiments\n"\
"    mozilla_firefox.pre_processing_experiments\n"\
"    mozilla_firefox.scrap_mozilla_firefox\n"\
"    mozilla_firefox.size_of_data_set_experiments\n"\
"    mozilla_firefox.tr_representation_experiments\n"\
"    mozilla_firefox.tuning_individual_classifiers_experiments\n"\
"\n"\
"Submodules\n"\
"----------\n"\
"\n"\
"mozilla\_firefox\.mozilla\_experiments\_launcher module\n"\
"-------------------------------------------------------\n"\
"\n"\
".. automodule:: mozilla_firefox.mozilla_experiments_launcher\n"\
"    :members:\n"\
"    :undoc-members:\n"\
"    :show-inheritance:\n\n" > ./doc/source/mozilla_firefox.rst

echo "..."

printf ".. Open-source software automatic bug assignment documentation master file, created by\n"\
"   sphinx-quickstart on Wed May 17 12:46:20 2017.\n"\
"   You can adapt this file completely to your liking, but it should at least\n"\
"   contain the root \`toctree\` directive.\n"\
"\n"\
"Welcome to Open-source software (OSS) automatic bug assignment's documentation!\n"\
"===============================================================================\n"\
"\n"\
".. toctree::\n"\
"   :maxdepth: 10\n"\
"   :caption: Contents:\n"\
"\n"\
"   modules\n"\
"\n"\
"\n"\
"Indices and tables\n"\
"==================\n"\
"\n"\
"* :ref:\`genindex\`\n"\
"* :ref:\`modindex\`\n"\
"* :ref:\`search\`\n" > ./doc/source/index.rst

read -p "Do you want to generate the HTML doc (y/n)?" response
if [ "$response" == "y" ] || [ "$response" == "Y" ]
then
    echo "Generating HTML documentation..."
    cd doc/
    make html
fi
