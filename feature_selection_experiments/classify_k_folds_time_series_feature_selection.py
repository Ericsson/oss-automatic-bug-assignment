# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import time
import numpy as np
import os
import inspect
import logging
import json

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir)
from utilities import print_log, load_data_set, build_data_frame
from scikit_learn.rfe import RFECV
from scikit_learn._search import GridSearchCV
from scikit_learn.accuracy_mrr_scoring_object import accuracy_mrr_scoring_object


class FeatureSelectionExperiment:
    def __init__(self, developers_dict_file=None, \
                developers_list_file=None):
        self._current_dir = None
        self._data_set_file = None
        self._developers_dict_file = developers_dict_file
        self._developers_list_file = developers_list_file   
        
        self._tscv = TimeSeriesSplit(n_splits=10) # Used to store a 
        # reference to the object which will allow us to perform a 
        # customized version of cross validation
        
        self._train_set = None # Used to store a reference to the 
        # training set
        self._test_set = None # Used to store a reference to the 
        # test set
        
        np.random.seed(0) # We set the seed
                
        self._pre_processing_steps = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=True, smooth_idf=False))]
                
        self._feature_selection_methods = [
            ("var_threshold", VarianceThreshold()),
            ("chi2", SelectPercentile(chi2)),
            ("f_classif", SelectPercentile(f_classif)),
            ("mutual_info_classif", SelectPercentile(mutual_info_classif))
        ]
        
        self._feature_selection_methods_params = [
            dict(var_threshold__threshold=[(((1-0)**2)/12)*i for i in np.arange(0.01,0.1,0.02)]),
            dict(chi2__percentile=list(range(10,100,20))),
            dict(f_classif__percentile=list(range(10,100,20))),
            dict(mutual_info_classif__percentile=list(range(10,100,20)))
        ]
        
        self._classifiers_estimators = { \
            "Linear SVM": [("clf", LinearSVC(random_state=0))], \
            # "MultinomialNB": [("clf", MultinomialNB())],
            # "LogisticRegression": [("clf", LogisticRegression(random_state=0, class_weight="balanced", multi_class="multinomial", n_jobs=-1))] \
        }

        # Below, there is a dictionary to store the names, the 
        # classifiers used, the parameters sent to the constructor of
        # the classifiers and the fitted classifiers    
        self._models_cv = {}
        
        # Below, there is a dictionary to store the names, the pipelines 
        # used, the parameters sent to the constructor of the feature 
        # selection techniques
        self._rfe_cv = {
            "RFECV SVM": [RFECV, {
                "estimator": LinearSVC(random_state=0),
                "step": 0.1,
                "cv": self._tscv,
                "scoring": accuracy_mrr_scoring_object, 
                "verbose": 10,
                "n_jobs": -1
            }, None], \
            # "RFECV Naive Bayes": [RFECV, {
            #     "estimator": MultinomialNB(),
            #     "step": 0.1,
            #     "cv": self._tscv,
            #     "verbose": 1,
            #     "n_jobs": -1
            # }, None]   
        }
        
        # Below, there is a dictionary used to save the cleaned 
        # results to a JSON file
        self._results_to_save_to_a_file = {}
        self._cleaned_results_file_name = "cleaned_feature_" + \
        "selection_experiment_results.json"
        
        for key, classifier_estimator in self._classifiers_estimators.items():
            for i, feature_selection_method in enumerate(self._feature_selection_methods):
                self._models_cv["GridSearch " + feature_selection_method[0] + " " + key] = [GridSearchCV, { \
                    "estimator": Pipeline(self._pre_processing_steps + [feature_selection_method] + classifier_estimator), \
                    "param_grid": self._feature_selection_methods_params[i], \
                    "n_jobs": -1, \
                    "iid": False, \
                    "cv": self._tscv, \
                    "verbose": 10, \
                    "error_score": np.array([-1, -1]), \
                    "scoring": accuracy_mrr_scoring_object
                }, None]
                
    def _build_data_set(self):
        # First we load the data of the three aforementioned files
        json_data = load_data_set(self._data_set_file)
        
        developers_dict_data = None
#         TO DO
#         load_developers_mappings(self._developers_dict_file)
        developers_list_data = None
#         TO DO
#         load_distinct_developers_list(self._developers_list_file)
        
        # Then, we build a data frame using the loaded data set, the 
        # loaded developers mappings, the loaded distinct developers.
        self._df = build_data_frame(json_data, developers_dict_data, \
                                    developers_list_data)
        
        print_log("Splitting the data set") # Debug
        # self._df = self._df[-30000:]
        self._train_set, self._test_set = np.split(self._df, \
        [int(.9*len(self._df))])
        print_log("Shape of the initial Data Frame") # Debug
        print_log(self._df.shape) # Debug
        print_log(self._df['class'].value_counts(normalize=True))
        print_log("Shape of the training set") # Debug
        print_log(self._train_set.shape) # Debug
        print_log(self._train_set['class'].value_counts(normalize=True))
        print_log("Shape of the test set") # Debug
        print_log(self._test_set.shape) # Debug
        print_log(self._test_set['class'].value_counts(normalize=True))
        
    def conduct_experiment(self):
        """Method used to conduct the experiment"""
        self._train_predict_cv()
        self._write_df()
        self._save_cleaned_results()
        
    def _save_cleaned_results(self):
        """Method to write the cleaned results

        It writes the cleaned results into a JSON file which path is 
        an attribute of the object.
        """
        with open(self._cleaned_results_file_name, 'w') as output_file:
            json.dump(self._results_to_save_to_a_file, output_file, \
                      indent=4)
        
    def _train_predict_cv(self):
        X_train = self._train_set['text']
        y_train = self._train_set['class'].values

        print_log("Training of the models") # Debug
        for key in self._models_cv:
            start_time = time.time() # We get the time expressed in 
            # seconds since the epoch
            print_log(key)
            self._models_cv[key][-1] = self. \
            _models_cv[key][0](**self._models_cv[key][1]) \
            .fit(X_train, y_train)
            print_log("--- {} seconds ---" \
                      .format(time.time() - start_time))

        self._results_to_save_to_a_file["avg_accuracy"] = {}
        self._results_to_save_to_a_file["mrr_accuracy"] = {}

        for key in self._models_cv:
            print_log("{}".format(key)) # Debug
            print_log("Best parameters set found on the training set:")
            print_log(self._models_cv[key][-1].best_params_)
            print_log("Grid scores on the training set:")
            means = self._models_cv[key][-1].cv_results_['mean_test_score']
            stds = self._models_cv[key][-1].cv_results_['std_test_score']
            self._results_to_save_to_a_file["avg_accuracy"][key] = \
            means[:,0].tolist()
            self._results_to_save_to_a_file["mrr_accuracy"][key] = \
            means[:,1].tolist()
#             print(means)
#             print(stds)
#             print(self._models_cv[key][-1].cv_results_['params'])
            for mean, std, params in zip(means, stds, self._models_cv[key][-1].cv_results_['params']):
                print_log("{} (+/-{}) for {!r}".format(mean, std * 2, params))
            print_log("All results on the training set")
            print_log(self._models_cv[key][-1].cv_results_)

        print_log("We count the occurrence of each term") # Debug
        count_vectorizer = CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")
        X_train_counts = count_vectorizer \
        .fit_transform(X_train)
        print_log(X_train_counts.shape)
        print_log("Use of the TF-IDF model") # Debug
        tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=False) 
        print_log("Computation of the weights of the TF-IDF model")
        X_train = tfidf_transformer.fit_transform(X_train_counts)
        
        print_log("Training of the models") # Debug
        for key in self._rfe_cv:
            start_time = time.time() # We get the time expressed in seconds 
            # since the epoch
            print_log(key)
            self._rfe_cv[key][-1] = \
            self._rfe_cv[key][0](**self._rfe_cv[key][1]) \
            .fit(X_train, y_train)
            print_log("Number of features")
            print_log(self._rfe_cv[key][-1].n_features_)
            print_log("The mask of selected features")
            print_log(self._rfe_cv[key][-1].support_)
            print_log("The feature ranking")
            print_log(self._rfe_cv[key][-1].ranking_)
            print_log("Cross validation scores")
            print_log(self._rfe_cv[key][-1].grid_scores_)
            self._results_to_save_to_a_file["avg_accuracy"][key] = \
            self._rfe_cv[key][-1].grid_scores_[:,0].tolist()
            self._results_to_save_to_a_file["mrr_accuracy"][key] = \
            self._rfe_cv[key][-1].grid_scores_[:,1].tolist()
            print_log("--- {} seconds ---" \
                      .format(time.time() - start_time)) 
    
    def _write_df(self):
        # We dump the data frame
        self._df.to_csv("pre_processed_data.csv")