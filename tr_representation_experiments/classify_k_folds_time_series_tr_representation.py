# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
# from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
import time
import numpy as np
import os
import inspect
import math
import logging
import json

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir) 
from utilities import load_data_set, build_data_frame, print_log
from scikit_learn._search import GridSearchCV
from scikit_learn.accuracy_mrr_scoring_object import accuracy_mrr_scoring_object

class TRRepresentationExperiment:

    def __init__(self, developers_dict_file=None, \
        developers_list_file=None):
        self._current_dir = None
        self._data_set_file = None
        self._developers_dict_file = developers_dict_file
        self._developers_list_file = developers_list_file      
        self._tscv = TimeSeriesSplit(n_splits=10)
        self._train_set = None
        self._test_set = None
                
        np.random.seed(0) # We set the seed
        
        # self._boolean_bernouilli = [("count", CountVectorizer( \
        # lowercase=False, token_pattern=u"(?u)\S+", binary=True)), \
        # ("clf", BernoulliNB()) \
        # ]
    
        self._boolean_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+", binary=True)), \
        ("clf", LinearSVC(random_state=0))
        ]
        
        self._tf_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=False, smooth_idf=False)), \
        ("clf", LinearSVC(random_state=0)) \
        ]
    
        self._tf_idf_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=True, smooth_idf=False)), \
        ("clf", LinearSVC(random_state=0)) \
        ]
            
        self._boolean_truncated_svd_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+", binary=True)), \
        ("truncated_svd", TruncatedSVD(n_components=100, random_state=0)), \
        ("normalizer", Normalizer(copy=False)), \
        ("clf", LinearSVC(random_state=0))
        ]
    
        self._tf_truncated_svd_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=False, smooth_idf=False)), \
        ("truncated_svd", TruncatedSVD(n_components=100, random_state=0)), \
        ("normalizer", Normalizer(copy=False)), \
        ("clf", LinearSVC(random_state=0))
        ]
    
        self._tf_idf_truncated_svd_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=True, smooth_idf=False)), \
        ("truncated_svd", TruncatedSVD(n_components=100, random_state=0)), \
        ("normalizer", Normalizer(copy=False)), \
        ("clf", LinearSVC(random_state=0))
        ]
    
        self._truncated_svd_params = dict(truncated_svd__n_components= \
        [math.floor(i/10*100) for i in range(1, 10, 2)])    
    
        self._boolean_nmf_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+", binary=True)), \
        ("nmf", NMF(n_components=100, random_state=0, alpha=.1, \
        l1_ratio=.5)), \
        ("clf", LinearSVC(random_state=0)) \
        ]

        self._tf_nmf_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=False, smooth_idf=False)), \
        ("nmf", NMF(n_components=100, random_state=0, alpha=.1, \
        l1_ratio=.5)), \
        ("clf", LinearSVC(random_state=0)) \
        ]
    
        self._tf_idf_nmf_svm = [("count", CountVectorizer( \
        lowercase=False, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=True, smooth_idf=False)), \
        ("nmf", NMF(n_components=100, random_state=0, alpha=.1, \
        l1_ratio=.5)), \
        ("clf", LinearSVC(random_state=0)) \
        ]    
    
        self._nmf_params = dict(nmf__n_components= \
        [math.floor(i/10*100) for i in range(1, 10, 2)])
    
        self._models_names_pipelines_mapping = {
            "Boolean SVM": self._boolean_svm[:-1],
            "TF SVM": self._tf_svm[:-1],
            "TF IDF SVM": self._tf_idf_svm[:-1],
            "GridSearch Boolean Truncated SVD SVM": self._boolean_truncated_svd_svm[:-1],
            "GridSearch TF Truncated SVD SVM": self._tf_truncated_svd_svm[:-1],
            "GridSearch TF IDF Truncated SVD SVM": self._tf_idf_truncated_svd_svm[:-1],
            "GridSearch Boolean NMF SVM": self._boolean_nmf_svm[:-1],
            "GridSearch TF NMF SVM": self._tf_nmf_svm[:-1],
            "GridSearch TF IDF NMF SVM": self._tf_idf_nmf_svm[:-1]
        }
    
        # Below, there is a dictionary to store the names, the 
        # classifiers used, the parameters sent to the constructor of
        # the classifiers and the fitted classifiers
        self._models = { \
            # "Boolean": [Pipeline, {
            #     "steps": self._boolean_bernouilli
            # }, None], \
            "Boolean SVM": [Pipeline, {
                "steps": self._boolean_svm
            }, None], \
            "TF SVM": [Pipeline, {
                "steps": self._tf_svm
            }, None], \
            "TF IDF SVM": [Pipeline, {
                "steps": self._tf_idf_svm
            }, None], \
            # "LDA": [Pipeline, {
            #     "steps": lda,
            # }, None]        
        }
    
        self._models_cv = { \
            "GridSearch Boolean Truncated SVD SVM": [GridSearchCV, {
                "estimator": Pipeline(self._boolean_truncated_svd_svm),
                "param_grid": self._truncated_svd_params,
                "n_jobs": 3,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "GridSearch TF Truncated SVD SVM": [GridSearchCV, {
                "estimator": Pipeline(self._tf_truncated_svd_svm),
                "param_grid": self._truncated_svd_params,
                "n_jobs": 3,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "GridSearch TF IDF Truncated SVD SVM": [GridSearchCV, {
                "estimator": Pipeline(self._tf_idf_truncated_svd_svm),
                "param_grid": self._truncated_svd_params,
                "n_jobs": 3,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "GridSearch Boolean NMF SVM": [GridSearchCV, {
                "estimator": Pipeline(self._boolean_nmf_svm),
                "param_grid": self._nmf_params,
                "n_jobs": 3,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "GridSearch TF NMF SVM": [GridSearchCV, {
                "estimator": Pipeline(self._tf_nmf_svm),
                "param_grid": self._nmf_params,
                "n_jobs": 3,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "GridSearch TF IDF NMF SVM": [GridSearchCV, {
                "estimator": Pipeline(self._tf_idf_nmf_svm),
                "param_grid": self._nmf_params,
                "n_jobs": 3,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
        }
        
        # Below, there is a dictionary to store the accuracies (per 
        # fold) of each configuration
        self._configurations_accuracies = {}
        # Below, there is a dictionary to store the MRR values (per 
        # fold) of each configuration
        self._configurations_mrr_values = {}
        
        self._combined_models = {}
                
        # Below, there is a dictionary to store the accuracies (per 
        # fold) of each combined configuration
        self._combined_configurations_accuracies = {}
        # Below, there is a dictionary to store the MRR values (per 
        # fold) of each combined configuration
        self._combined_configurations_mrr_values = {}
        
        # Below, there is a dictionary used to save the cleaned 
        # results to a JSON file
        self._results_to_save_to_a_file = {}
        self._cleaned_results_file_name = "cleaned_tr_" + \
        "representation_experiment_results.json"
        
    def conduct_experiment(self):
        self._train_predict(self._models, \
                            self._configurations_accuracies, \
                            self._configurations_mrr_values, \
                            False)
        self._train_predict_cv()
        self._update_models_names_pipelines_mapping()
        self._build_combined_pipelines()
        self._train_predict(self._combined_models, \
                            self._combined_configurations_accuracies, \
                            self._combined_configurations_mrr_values, \
                            True)
        self._write_df()
        self._save_cleaned_results()
        
    def _save_cleaned_results(self):
        """Method to write the cleaned results

        It writes the cleaned results into a JSON file which path is 
        an attribute of the object.
        """
        with open(self._cleaned_results_file_name, 'w') as output_file:
            json.dump(self._results_to_save_to_a_file, output_file)
        
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

    def _train_predict(self, models, models_accuracies, models_mrrs, \
                       combined=False):
        start_time = time.time() # We get the time expressed in 
        # seconds since the epoch
        i = 1 # Used to know on each fold we will test the classifier 
        # currently trained 
        for train_indices, val_indices in self._tscv.split(self._train_set):
            print_log("********* Evaluation on fold {} *********" \
            .format(i)) # Debug

            X_train = self._train_set.iloc[train_indices]["text"]
            y_train = self._train_set.iloc[train_indices]["class"].values
       
            print_log("Training of the models") # Debug
            for key, value in models.items():
                print_log(key)
                models[key][-1] = models[key][0](**models[key][1]) \
                .fit(X_train, y_train)         
                print_log("--- {} seconds ---" \
                .format(time.time() - start_time))
            
            X_val = self._train_set.iloc[val_indices]["text"].values
            y_val = self._train_set.iloc[val_indices]["class"].values
        
            print_log("Making predictions") # Debug
            
            for key, value in models.items():
                if i == 1:
                    models_accuracies[key] = []
                    models_mrrs[key] = []
                models_accuracies[key].append(\
                np.mean(value[-1].predict(X_val) == y_val))

                found_function = False                        
                try:
                    if callable(getattr(value[-1], "predict_proba")):
#                         print_log(value[-1].classes_)
#                         print_log(value[-1].predict_proba(X_val))
                        lb = LabelBinarizer()        
                        _ = lb.fit_transform(value[-1].classes_)
#                         print_log(lb.classes_)
#                         print_log(y_classes_bin)
#                         print_log(lb.transform(["X"]))  
                        y_val_bin = lb.transform(y_val)
                        models_mrrs[key].append(\
                        label_ranking_average_precision_score( \
                        y_val_bin, \
                        value[-1].predict_proba(X_val)))
                        found_function = True
                except AttributeError:
                    pass
                
                try:
                    if not found_function and callable(getattr(value[-1], "decision_function")):
#                         print_log(value[-1].classes_)
#                         print_log(value[-1].decision_function(X_val))
                        lb = LabelBinarizer()        
                        _ = lb.fit_transform(value[-1].classes_)
#                         print_log(lb.classes_)
#                         print_log(y_classes_bin)
#                         print_log(lb.transform(["X"]))  
                        y_val_bin = lb.transform(y_val)
                        models_mrrs[key].append(\
                        label_ranking_average_precision_score( \
                        y_val_bin, \
                        value[-1].decision_function(X_val)))
                        found_function = True
                except AttributeError:
                    pass
                print_log("Mean Reciprocal Rank:") # Debug
                print_log(models_mrrs[key][-1]) # Debug 
                print_log("--- {} seconds ---" \
                          .format(time.time() - start_time))               

            i += 1
            
        if combined:
            self._results_to_save_to_a_file["combined_avg_accuracy"] = {}
            self._results_to_save_to_a_file["combined_avg_mrr"] = {} 
        else:
            self._results_to_save_to_a_file["not_combined_avg_accuracy"] = {}
            self._results_to_save_to_a_file["not_combined_avg_mrr"] = {}
        
        avg_accuracy = None
        avg_mrr = None
            
        # Below, we print the average accuracies
        for key, value in models_accuracies.items():
            print_log("Accuracy of {}".format(key)) # Debug
            print_log("Each fold")
            print_log(value)
            print_log("Average")
            avg_accuracy = sum(value)/len(value)
            if combined:
                self._results_to_save_to_a_file["combined_avg_accuracy"][key] = \
                avg_accuracy
            else:
                self._results_to_save_to_a_file["not_combined_avg_accuracy"][key] = \
                avg_accuracy
            
            print_log(avg_accuracy) # Debug
            
            print_log("MRR of {}".format(key)) # Debug
            print_log("Each fold")
            print_log(models_mrrs[key])
            print_log("Average")
            avg_mrr = sum(models_mrrs[key])/len(models_mrrs[key])
            if combined:
                self._results_to_save_to_a_file["combined_avg_mrr"][key] = \
                avg_mrr
            else:
                self._results_to_save_to_a_file["not_combined_avg_mrr"][key] = \
                avg_mrr
            
            print_log(avg_mrr) # Debug        

        print_log("--- {} seconds ---" \
                  .format(time.time() - start_time))
        
    def _train_predict_cv(self):
        print_log("Training of the models") # Debug
        X_train = self._train_set['text']
        y_train = self._train_set['class'].values
        for key in self._models_cv:
            start_time = time.time() # We get the time expressed in 
            # seconds since the epoch
            print_log(key)
            self._models_cv[key][-1] = \
            self._models_cv[key][0](**self._models_cv[key][1]) \
            .fit(X_train, y_train)         
            print_log("--- {} seconds ---" \
                      .format(time.time() - start_time))
        
        for key in self._models_cv:
            print_log("{}".format(key)) # Debug    
            print_log("Best parameters set found on the training set:")
            print_log(self._models_cv[key][-1].best_params_)
            print_log("Grid scores on the training set:")
            means = self._models_cv[key][-1].cv_results_['mean_test_score']
            stds = self._models_cv[key][-1].cv_results_['std_test_score']
            self._results_to_save_to_a_file["not_combined_avg_accuracy"][key] = \
            means[:,0].tolist()
            self._results_to_save_to_a_file["not_combined_avg_mrr"][key] = \
            means[:,1].tolist()
            for mean, std, params in zip(means, stds, self._models_cv[key][-1].cv_results_['params']):
                print_log("{} (+/-{}) for {!r}" \
                          .format(mean, std * 2, params))
            print_log("All results on the training set")
            print_log(self._models_cv[key][-1].cv_results_)
            
    def _update_models_names_pipelines_mapping(self):
        for key in self._models_names_pipelines_mapping:        
            try:
                self._models_cv[key]
                if "truncated_svd__n_components" in self._models_cv[key][-1].best_params_:
                    best_n_components = \
                    self._models_cv[key][-1]\
                    .best_params_["truncated_svd__n_components"]
                    self._models_names_pipelines_mapping[key][-2] = \
                    ("truncated_svd", \
                     TruncatedSVD(n_components=best_n_components, \
                                  random_state=0))  
                else:
                    best_n_components = \
                    self._models_cv[key][-1]. \
                    best_params_["nmf__n_components"]
                    self._models_names_pipelines_mapping[key][-1] = \
                    ("nmf", \
                     NMF(n_components=best_n_components, \
                         random_state=0, alpha=.1, l1_ratio=.5))
            except KeyError:
                # We don't need to find the best parameters of the 
                # model
                pass
                
    def _build_combined_pipelines(self):
        keys = list(self._models_names_pipelines_mapping.keys())
        for i in range(len(keys)-1):
            for j in range(i+1, len(keys)):
                self._combined_models[keys[i] + "_" + keys[j]] = [ \
                    Pipeline, { \
                        "steps": [ \
                            ("features", FeatureUnion(transformer_list=[ \
                                ("features_1", Pipeline(self._models_names_pipelines_mapping[keys[i]])), \
                                ("features_2", Pipeline(self._models_names_pipelines_mapping[keys[j]])) \
                            ])), \
                            ("clf", LinearSVC(random_state=0)) \
                        ] \
                }, None]

    def _write_df(self):
        # We dump the data frame
        self._df.to_csv("pre_processed_data.csv")
        self._models_names_pipelines_mapping = {
            "Boolean SVM": self._boolean_svm[:-1],
            "TF SVM": self._tf_svm[:-1],
            "TF IDF SVM": self._tf_idf_svm[:-1],
            "GridSearch Boolean Truncated SVD SVM": self._boolean_truncated_svd_svm[:-1],
            "GridSearch TF Truncated SVD SVM": self._tf_truncated_svd_svm[:-1],
            "GridSearch TF IDF Truncated SVD SVM": self._tf_idf_truncated_svd_svm[:-1],
            "GridSearch Boolean NMF SVM": self._boolean_nmf_svm[:-1],
            "GridSearch TF NMF SVM": self._tf_nmf_svm[:-1],
            "GridSearch TF IDF NMF SVM": self._tf_idf_nmf_svm[:-1]
        }
       
        self._configurations_accuracies = {}
        self._configurations_mrr_values = {}
        self._combined_models = {}
        self._combined_configurations_accuracies = {}
        self._combined_configurations_mrr_values = {}
        for key in self._models:
            self._models[key][2] = None
        for key in self._models_cv:
            self._models_cv[key][2] = None