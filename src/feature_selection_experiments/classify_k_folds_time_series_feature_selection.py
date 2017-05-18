# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import time
import numpy as np
import os
import inspect
import logging
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0,parent_dir)
from experiment import Experiment
from utilities import print_log
from scikit_learn.rfe import RFECV
from scikit_learn._search import GridSearchCV
from scikit_learn.accuracy_mrr_scoring_object import accuracy_mrr_scoring_object

class FeatureSelectionExperiment(Experiment):
    
    @abc.abstractmethod
    def __init__(self, data_set_file, lowercase=False, use_idf=False,
                 developers_dict_file=None, 
                 developers_list_file=None):
        super().__init__(developers_dict_file, developers_list_file)
        np.random.seed(0) # We set the seed
        self.lowercase = lowercase
        self.use_idf = use_idf
        
        self._pre_processing_steps = [("count", CountVectorizer( \
        lowercase=lowercase, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=use_idf, smooth_idf=False))]
                
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
                
        cleaned_results_file_name = "cleaned_feature_selection_" + \
        "experiment_results.json"
        self._cleaned_results_file_name = os.path.join( \
        self._current_dir, cleaned_results_file_name)
                
        self._data_set_file = os.path.join(self._current_dir, \
        data_set_file)
        
        log_file = os.path.join(self._current_dir, \
                                "feature_selection_experiment.log")
        logging.basicConfig(filename=log_file, filemode="w", \
                            level=logging.DEBUG)
        
        self._build_data_set()
       
    def conduct_experiment(self):
        """Method used to conduct the experiment"""
        self._train_predict_cv()
        self._write_df()
        super().conduct_experiment()
        
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
        self._results_to_save_to_a_file["avg_mrr"] = {}

        for key in self._models_cv:
            print_log("{}".format(key)) # Debug
            print_log("Best parameters set found on the training set:")
            print_log(self._models_cv[key][-1].best_params_)
            print_log("Grid scores on the training set:")
            means = self._models_cv[key][-1].cv_results_['mean_test_score']
            stds = self._models_cv[key][-1].cv_results_['std_test_score']
            self._results_to_save_to_a_file["avg_accuracy"][key] = \
            means[:,0].tolist()
            self._results_to_save_to_a_file["avg_mrr"][key] = \
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
        lowercase=self.lowercase, token_pattern=u"(?u)\S+")
        X_train_counts = count_vectorizer \
        .fit_transform(X_train)
        print_log(X_train_counts.shape)
        print_log("Use of the TF-IDF model") # Debug
        tfidf_transformer = TfidfTransformer(use_idf=self.use_idf, smooth_idf=False) 
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
            self._results_to_save_to_a_file["avg_mrr"][key] = \
            self._rfe_cv[key][-1].grid_scores_[:,1].tolist()
            print_log("--- {} seconds ---" \
                      .format(time.time() - start_time))