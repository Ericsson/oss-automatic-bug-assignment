# -*- coding: utf-8 -*-
"""
.. module:: classify_k_folds_time_series_tuning
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to conduct 
              the last experiment of the thesis. The experiment 
              consists mainly of tuning several classifiers and 
              selecting the best performing one.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from scipy.stats import uniform as uni
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
import time
import numpy as np
import os
import inspect
import logging
import abc

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
from log_space_uniform import LogSpaceUniform
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from exp_launcher import ExpLauncher
from utilities import print_log
from scikit_learn._search import GridSearchCV, RandomizedSearchCV
from scikit_learn.accuracy_mrr_scoring_object \
import accuracy_mrr_scoring_object

class Exp4Launcher(ExpLauncher):
    
    @abc.abstractmethod
    def __init__(self, data_set_file, lowercase=False, use_idf=False, 
                 developers_dict_file=None, 
                 developers_list_file=None):
        super().__init__(developers_dict_file, developers_list_file)
        np.random.seed(0) # We set the seed
        self.lowercase = lowercase
        self.use_idf = use_idf
        
        self._estimators = [("count", CountVectorizer( \
        lowercase=self.lowercase, token_pattern=u"(?u)\S+")), \
        ("tf_idf", TfidfTransformer(use_idf=self.use_idf, smooth_idf=False))]
                
        self._nearest_centroid_estimators = self._estimators + \
        [("clf", NearestCentroid())]
        self._naive_bayes_estimators = self._estimators + \
        [("clf", MultinomialNB())]
        self._linear_svm_estimators = self._estimators + \
        [("clf", LinearSVC())]
        self._logistic_regression_estimators = self._estimators + \
        [("clf", LogisticRegression())]
        self._perceptron_estimators = self._estimators + \
        [("clf", Perceptron())]
        self._stochastic_gradient_descent_estimators = \
        self._estimators + [("clf", SGDClassifier())]

        self._nearest_centroid_estimators_params = \
        dict( \
            clf__metric=["manhattan", "euclidean"] \
        )        
        self._naive_bayes_estimators_params = \
        dict( \
            clf__alpha=np.linspace(0, 1, 11), \
            clf__fit_prior=[True, False] \
        )
        self._linear_svm_estimators_params = \
        dict( \
            clf__C=np.logspace(-4, 4, 10), \
            clf__loss=["squared_hinge", "hinge"], \
            clf__class_weight=["balanced"] \
        )
        self._primal_logistic_regression_estimators_params = \
        dict( \
            clf__dual=[False],
            clf__C=np.logspace(-4, 4, 10), \
            clf__class_weight=["balanced"], \
            clf__solver=["newton-cg", "sag", "lbfgs"], \
            clf__multi_class=["multinomial"]
        )
        self._dual_logistic_regression_estimators_params = \
        dict( \
            clf__dual=[True],
            clf__C=np.logspace(-4, 4, 10), \
            clf__class_weight=["balanced"], \
            clf__solver=["liblinear"], \
            clf__multi_class=["ovr"]
        )
        self._perceptron_with_penalty_estimators_params = \
        dict( \
            clf__penalty=["l2", "elasticnet"],
            clf__alpha=10.0**-np.arange(1,7), \
            clf__class_weight=["balanced"]
        )
        self._perceptron_without_penalty_estimators_params = \
        dict( \
            clf__penalty=[None],
            clf__class_weight=["balanced"]
        )        
        self._stochastic_gradient_descent_estimators_params = \
        dict( \
            clf__loss=["hinge", "log", "modified_huber", \
            "squared_hinge", "perceptron"], \
            clf__penalty=["l2", "elasticnet"], \
            clf__alpha=10.0**-np.arange(1,7), \
            clf__class_weight=["balanced"], \
            clf__average=[True, False]
        )
        
        # Below, there is a dictionary to store the names, the 
        # classifiers used, the parameters sent to the constructor of
        # the classifiers and the fitted classifiers (grid search)
        self._models_cv = { \
            "NearestCentroid": [GridSearchCV, {
                "estimator": Pipeline(self._nearest_centroid_estimators),
                "param_grid": self._nearest_centroid_estimators_params,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "MultinomialNB": [GridSearchCV, {
                "estimator": Pipeline(self._naive_bayes_estimators),
                "param_grid": self._naive_bayes_estimators_params,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "LinearSVC": [GridSearchCV, {
                "estimator": Pipeline(self._linear_svm_estimators),
                "param_grid": self._linear_svm_estimators_params,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "Primal LogisticRegression": [GridSearchCV, {
                "estimator": Pipeline(self._logistic_regression_estimators),
                "param_grid": self._primal_logistic_regression_estimators_params,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "Dual LogisticRegression": [GridSearchCV, {
                "estimator": Pipeline(self._logistic_regression_estimators),
                "param_grid": self._dual_logistic_regression_estimators_params,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "PerceptronWithPenalty": [GridSearchCV, {
                "estimator": Pipeline(self._perceptron_estimators),
                "param_grid": self._perceptron_with_penalty_estimators_params,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "PerceptronWithoutPenalty": [GridSearchCV, {
                "estimator": Pipeline(self._perceptron_estimators),
                "param_grid": self._perceptron_without_penalty_estimators_params,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "SGDClassifier": [GridSearchCV, {
                "estimator": Pipeline(self._stochastic_gradient_descent_estimators),
                "param_grid": self._stochastic_gradient_descent_estimators_params,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
        }
        
        self._nearest_centroid_estimators_random_params = \
        dict( \
            clf__metric=["manhattan", "euclidean"] \
        )        
        self._naive_bayes_estimators_random_params = \
        dict( \
            clf__alpha=uni(loc=0,scale=1), \
            clf__fit_prior=[True, False] \
        )
        self._linear_svm_estimators_random_params = \
        dict( \
            clf__C=LogSpaceUniform(loc=-4,scale=8), \
            clf__loss=["squared_hinge", "hinge"], \
            clf__class_weight=["balanced"] \
        )
        self._primal_logistic_regression_estimators_random_params = \
        dict( \
            clf__dual=[False],
            clf__C=LogSpaceUniform(loc=-4,scale=8), \
            clf__class_weight=["balanced"], \
            clf__solver=["newton-cg", "sag", "lbfgs"], \
            clf__multi_class=["multinomial"]
        )
        self._dual_logistic_regression_estimators_random_params = \
        dict( \
            clf__dual=[True],
            clf__C=LogSpaceUniform(loc=-4,scale=8), \
            clf__class_weight=["balanced"], \
            clf__solver=["liblinear"], \
            clf__multi_class=["ovr"]
        )
        self._perceptron_with_penalty_estimators_random_params = \
        dict( \
            clf__penalty=["l2", "elasticnet"],
            clf__alpha=LogSpaceUniform(loc=-6,scale=5), \
            clf__class_weight=["balanced"]
        )
        self._perceptron_without_penalty_estimators_random_params = \
        dict( \
            clf__penalty=[None],
            clf__class_weight=["balanced"]
        )
        
        self._stochastic_gradient_descent_estimators_random_params = \
        dict( \
            clf__loss=["hinge", "log", "modified_huber", \
            "squared_hinge", "perceptron"], \
            clf__penalty=["l2", "elasticnet"], \
            clf__alpha=LogSpaceUniform(loc=-6,scale=5), \
            clf__class_weight=["balanced"], \
            clf__average=[True, False]
        )
        
        # Below, there is a dictionary to store the names, the 
        # classifiers used, the parameters sent to the constructor of 
        # the classifiers and the fitted classifiers (random search)
        self._randomized_models_cv = { \
            "NearestCentroid": [RandomizedSearchCV, {
                "estimator": Pipeline(self._nearest_centroid_estimators),
                "param_distributions": self._nearest_centroid_estimators_random_params,
                "n_iter": 2,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "MultinomialNB": [RandomizedSearchCV, {
                "estimator": Pipeline(self._naive_bayes_estimators),
                "param_distributions": self._naive_bayes_estimators_random_params,
                "n_iter": 22,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "LinearSVC": [RandomizedSearchCV, {
                "estimator": Pipeline(self._linear_svm_estimators),
                "param_distributions": self._linear_svm_estimators_random_params,
                "n_iter": 20,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "Primal LogisticRegression": [RandomizedSearchCV, {
                "estimator": Pipeline(self._logistic_regression_estimators),
                "param_distributions": self._primal_logistic_regression_estimators_random_params,
                "n_iter": 30,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "Dual LogisticRegression": [RandomizedSearchCV, {
                "estimator": Pipeline(self._logistic_regression_estimators),
                "param_distributions": self._dual_logistic_regression_estimators_random_params,
                "n_iter": 10,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "PerceptronWithPenalty": [RandomizedSearchCV, {
                "estimator": Pipeline(self._perceptron_estimators),
                "param_distributions": self._perceptron_with_penalty_estimators_random_params,
                "n_iter": 12,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "PerceptronWithoutPenalty": [RandomizedSearchCV, {
                "estimator": Pipeline(self._perceptron_estimators),
                "param_distributions": self._perceptron_without_penalty_estimators_random_params,
                "n_iter": 1,
                "n_jobs": 8,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
            "SGDClassifier": [RandomizedSearchCV, {
                "estimator": Pipeline(self._stochastic_gradient_descent_estimators),
                "param_distributions": self._stochastic_gradient_descent_estimators_random_params,
                "n_iter": 120,
                "n_jobs": 1,
                "iid": False,
                "cv": self._tscv,
                "verbose": 10,
                "random_state": 0,
                "error_score": np.array([-1, -1]),
                "scoring": accuracy_mrr_scoring_object
            }, None], \
        }
        
        # Below, there is a dictionary to store the accuracy of each 
        # configuration on the test set (grid search)
        self._configurations_accuracies = {}
        # Below, there is a dictionary to store the MRR value of each
        # configuration on the test set (grid search)
        self._configurations_mrr_values = {}

        # Below, there is a dictionary to store the accuracy of each 
        # configuration on the test set (random search)
        self._randomized_configurations_accuracies = {}
        # Below, there is a dictionary to store the MRR value of each
        # configuration on the test set (random search)
        self._randomized_configurations_mrr_values = {}
        
        cleaned_results_file_name = "cleaned_tuning_" + \
        "individual_classifier_generic_experiment_results.json"
        self._cleaned_results_file_name = os.path.join( \
        self._current_dir, cleaned_results_file_name)
        
        self._data_set_file = os.path.join(self._current_dir, \
        data_set_file)
        
        log_file = os.path.join(self._current_dir, \
        "tuning_individual_classifier_generic_experiment.log")
        logging.basicConfig(filename=log_file, filemode="w", \
                            level=logging.DEBUG)
        
        self._build_data_set()
               
    def _train_predict_cv(self, model_cv, random=False):
        print_log("Training of the models") # Debug
        X_train = self._train_set['text']
        y_train = self._train_set['class'].values
        for key in model_cv:
            start_time = time.time() # We get the time expressed in 
            # seconds since the epoch
            print_log(key)
            model_cv[key][-1] = model_cv[key][0](**model_cv[key][1]) \
            .fit(X_train, y_train)         
            print_log("--- {} seconds ---" \
                      .format(time.time() - start_time))

        if random is True:
            self._results_to_save_to_a_file["random_avg"] = {}
        else:
            self._results_to_save_to_a_file["normal_avg"] = {}
            
        for key in model_cv:
            print_log("{}".format(key)) # Debug    
            print_log("Best parameters set found on the training set:")
            print_log(model_cv[key][-1].best_params_)
            means = model_cv[key][-1].cv_results_['mean_test_score']
            stds = model_cv[key][-1].cv_results_['std_test_score']
            params_list = model_cv[key][-1].cv_results_['params']
            if random is True:
                self._results_to_save_to_a_file["random_avg"][key] = \
                {}
                self._results_to_save_to_a_file["random_avg"][key]["means_accuracy"] = \
                means[:,0].tolist()
                self._results_to_save_to_a_file["random_avg"][key]["means_mrr"] = \
                means[:,1].tolist()
                self._results_to_save_to_a_file["random_avg"][key]["params"] = \
                params_list
            else:
                self._results_to_save_to_a_file["normal_avg"][key] = \
                {}
                self._results_to_save_to_a_file["normal_avg"][key]["means_accuracy"] = \
                means[:,0].tolist()
                self._results_to_save_to_a_file["normal_avg"][key]["means_mrr"] = \
                means[:,1].tolist()
                self._results_to_save_to_a_file["normal_avg"][key]["params"] = \
                params_list
            print_log("Grid scores on the training set:")
            for mean, std, params in zip(means, stds, params_list):
                print_log("{} (+/-{}) for {!r}".format(mean, std * 2, params))
            print_log("All results on the training set")
            print_log(model_cv[key][-1].cv_results_)
        
    def _predict_test_set(self, model_cv, models_accuracies, \
                          models_mrrs):
        print_log("We count the occurrence of each term in the " + \
                  "test set") # Debug
        X_test = self._test_set['text'].values
        y_test = self._test_set['class'].values
        
        start_time = time.time() # We get the time expressed in 
        # seconds since the epoch
        
        print_log("Making predictions") # Debug
        
        for key in model_cv:
            start_time = time.time() # We get the time expressed in 
            # seconds since the epoch
            print_log(key)
            models_accuracies[key] = np.mean( \
            model_cv[key][-1].predict(X_test) == y_test)            
            found_function = False                        
            try:
                if callable(getattr(model_cv[key][-1], "predict_proba")):
#                     print_log(model_cv[key][-1].classes_)
#                     print_log(model_cv[key][-1].predict_proba(X_val))
                    lb = LabelBinarizer()        
                    _ = lb.fit_transform(model_cv[key][-1].classes_)
#                     print_log(lb.classes_)
#                     print_log(y_classes_bin)
#                     print_log(lb.transform(["X"]))  
                    y_test_bin = lb.transform(y_test)
                    models_mrrs[key] = \
                    label_ranking_average_precision_score( \
                    y_test_bin, \
                    model_cv[key][-1].predict_proba(X_test))
                    found_function = True
            except AttributeError:
                pass
            
            try:
                if not found_function and callable(getattr(model_cv[key][-1], "decision_function")):
#                     print_log(model_cv[key][-1].classes_)
#                     print_log(model_cv[key][-1].decision_function(X_val))
                    lb = LabelBinarizer()        
                    _ = lb.fit_transform(model_cv[key][-1].classes_)
#                     print_log(lb.classes_)
#                     print_log(y_classes_bin)
#                     print_log(lb.transform(["X"]))  
                    y_test_bin = lb.transform(y_test)
                    models_mrrs[key] = \
                    label_ranking_average_precision_score( \
                    y_test_bin, \
                    model_cv[key][-1].decision_function(X_test))
                    found_function = True
            except AttributeError:
                models_mrrs[key] = -1
            
            print_log("--- {} seconds ---" \
                      .format(time.time() - start_time))
            
        # Below, we print the accuracy of each classifier
        for key, value in models_accuracies.items():
            print_log("Accuracy of {}".format(key)) # Debug
            print_log(value) # Debug
            print_log("MRR of {}".format(key)) # Debug
            print_log(models_mrrs[key]) # Debug
        
    def conduct_experiment(self):
        """Method used to conduct the experiment"""
        print_log("### Grid Search ###") # Debug
        self._train_predict_cv(self._models_cv, False)
        self._predict_test_set(self._models_cv, \
                               self._configurations_accuracies, \
                               self._configurations_mrr_values)
        print_log("### Random Search ###") # Debug
        self._train_predict_cv(self._randomized_models_cv, True)
        self._predict_test_set(self._randomized_models_cv, \
                               self._randomized_configurations_accuracies, \
                               self._randomized_configurations_mrr_values)
        self._write_df()
        super().conduct_experiment()