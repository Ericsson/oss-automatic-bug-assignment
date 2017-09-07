# -*- coding: utf-8 -*-
"""
.. module:: size_of_data_set_experiment
   :platform: Unix, Windows
   :synopsis: This module contains an abstract class used to conduct 
              one of the two sub experiments of the preliminary 
              experiment of the thesis. The experiment consists mainly
              of trying to find the optimal number of bug reports that 
              should be used to train a classifier.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
import time
import matplotlib.pyplot as plt
import numpy as np
import abc
import os
import inspect

current_dir = os.path.dirname(os.path.abspath( \
inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
os.sys.path.insert(0, parent_dir)
from utilities import print_log, load_data_set, build_data_frame

class SizeOfDataExperiment(abc.ABC):

    @abc.abstractmethod
    def __init__(self, data_set_file, developers_dict_file=None, \
                 developers_list_file=None):
        """Constructor"""
        self._data_set_file = os.path.join(self._current_dir, \
        data_set_file)
        self._developers_dict_file = developers_dict_file
        self._developers_list_file = developers_list_file
        self._build_data_set()
        self._type = ""
        
    def _custom_linspace(self, start, stop, num):
        if num < 3:
            raise ValueError("num must be greater than 2")
        num -= 1
        size_of_fold = (stop - start) // num
        output = []
        for i in range(num):
            output.append(start + i * size_of_fold)
        output.append(stop)
        return output
    
    def _plot_learning_curve(self, title, computed_score, \
                             train_sizes, train_scores_mean, \
                             train_scores_std, test_scores_mean, \
                             test_scores_std, ylim=None):
        """Generate a plot of the test and training learning curves.

        Parameters
        ----------
        title: string
            Contains the title of the chart.

        computed_score: string
            Contains the name of the computed score.

        train_sizes: a one dimension numpy.ndarray
            An array containing the various sizes of the training set for 
            which the scores have been computed.

        train_scores_mean: a one dimension numpy.ndarray
            An array containing the various means of the scores related 
            to each element in train_sizes. These scores should have been 
            computed on the training set.

        train_scores_std: a one dimension numpy.ndarray
            An array containing the various standard deviations of the 
            scores related to each element in train_sizes. These scores 
            should have been computed on the training set.

        test_scores_mean: a one dimension numpy.ndarray
            An array containing the various means of the scores related 
            to each element in train_sizes. These scores should have been 
            computed on the test set.

        test_scores_std: a one dimension numpy.ndarray
            An array containing the various standard deviations of the 
            scores related to each element in train_sizes. These scores 
            should have been computed on the test set.

        ylim: tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        """
        fig = plt.figure(figsize=(20.0, 12.5))
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)

        plt.xlabel("Training examples")
        plt.ylabel(computed_score.capitalize())
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - \
            train_scores_std, train_scores_mean + train_scores_std, \
            alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - \
            test_scores_std, test_scores_mean + test_scores_std, \
            alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", \
               label="Training {}".format(computed_score))
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", \
               label="Test {}".format(computed_score))
        plt.legend(loc="best")
        
        return fig

    def _compute_mean_std_for_train_test_sets(self, train_indices, \
        train_scores, test_scores):
        """Computes training and test set accuracies for each size.

        Parameters
        ----------    
        train_indices: a list of lists
            Each element of the top-list is a list containing the 
            training set indices for which two scores have been computed
            (one for the training set and one for the test set)

        train_scores: a two dimensions numpy.ndarray, shape(1, n_exps)
            Each element at the index i of train_scores[0] contains the 
            computed score related to the training set indices 
            train_indices[i]. This score has been computed for the above 
            mentioned training set.

        test_scores: a two dimensions numpy.ndarray, shape(1, n_exps)
            Each element at the index i of train_scores[0] contains the 
            computed score related to the training set indices 
            train_indices[i]. This score has been computed for a test set
            which indices follow the ones in train_indices[i].

        Returns
        -------
        (train_sizes, train_scores_mean, train_scores_std, 
        test_scores_mean, test_scores_std): a tuple of five one dimension 
        numpy.ndarray
            Each element of this tuple is described below.

        train_sizes: a one dimension numpy.ndarray
            An array containing the various sizes of the training set for 
            which the scores have been computed.

        train_scores_mean: a one dimension numpy.ndarray
            An array containing the various means of the scores related 
            to each element in train_sizes. These scores should have been 
            computed on the training set.

        train_scores_std: a one dimension numpy.ndarray
            An array containing the various standard deviations of the 
            scores related to each element in train_sizes. These scores 
            should have been computed on the training set.

        test_scores_mean: a one dimension numpy.ndarray
            An array containing the various means of the scores related 
            to each element in train_sizes. These scores should have been 
            computed on the test set.

        test_scores_std: a one dimension numpy.ndarray
            An array containing the various standard deviations of the 
            scores related to each element in train_sizes. These scores 
            should have been computed on the test set.
        """

        # The lines below are used for debugging purpose
        print_log("len(train_indices): {}".format(len(train_indices)))
        print_log("len(train_indices): {}".format(len(train_scores[0])))
        print_log("len(train_indices): {}".format(len(test_scores[0])))

        # Dictionaries used to store the lists containing all accuracies
        # computed for each training set size. One dictionary is used to 
        # store the accuracies related to the training set. The other one
        # is used to store the accuracies related to the test set.
        train_accuracy_per_size = {}
        test_accuracy_per_size = {}

        # Below, the aforementioned dictionaries are filled.
        for i, train_indice in enumerate(train_indices):
            try:
                train_accuracy_per_size[len(train_indice)] \
                .append(train_scores[0][i])
            except KeyError:
                train_accuracy_per_size[len(train_indice)] = \
                [train_scores[0][i]]
            try:
                test_accuracy_per_size[len(train_indice)] \
                .append(test_scores[0][i])
            except KeyError:   
                test_accuracy_per_size[len(train_indice)] = \
                [test_scores[0][i]]

        print_log("train_accuracy_per_size: {}"\
            .format(train_accuracy_per_size))
        print_log("test_accuracy_per_size: {}"\
            .format(test_accuracy_per_size))

        train_sizes = [] # Will contain the training set sizes

        # The lists below will be used to store the aforementioned means
        # and standard deviations 
        train_scores_mean = []
        train_scores_std = []
        test_scores_mean = []
        test_scores_std = []
        
        # The aforementioned lists are filled
        for key in sorted(train_accuracy_per_size):
            train_sizes.append(key)
            train_scores_mean.append(np.mean(train_accuracy_per_size[key]))      
            train_scores_std.append(np.std(train_accuracy_per_size[key]))
            test_scores_mean.append(np.mean(test_accuracy_per_size[key]))
            test_scores_std.append(np.std(test_accuracy_per_size[key]))

        # Finally, these lists are converted to some one dimension 
        # numpy.ndarray
        train_sizes = np.asarray(train_sizes)
        train_scores_mean = np.asarray(train_scores_mean)
        train_scores_std = np.asarray(train_scores_std)
        test_scores_mean = np.asarray(test_scores_mean)
        test_scores_std = np.asarray(test_scores_std)
        
        # The lines below are used for debugging purpose
        print_log("train_sizes: {}".format(train_sizes))
        print_log("train_scores_mean: {}".format(train_scores_mean))
        print_log("train_scores_std: {}".format(train_scores_std))
        print_log("test_scores_mean: {}".format(test_scores_mean))
        print_log("test_scores_std: {}".format(test_scores_std))    

        return (train_sizes, train_scores_mean, train_scores_std, \
            test_scores_mean, test_scores_std)
        
    @abc.abstractmethod
    def _yield_indices_for_learning_curve(self, K=33):
        pass
    
    @abc.abstractmethod
    def _generate_list_indices_for_learning_curve(self, K=33):
        pass
    
    @abc.abstractmethod
    def plot_or_save_learning_curve(self, K=33, save_file=False):
        start_time = time.time() # We get the time expressed in 
        # seconds since the epoch
        # The lines below are used for debugging purpose
        for train, test in self._yield_indices_for_learning_curve(K):
            print_log("*************")
            print_log("{} - {} / {} - {}".format(train[0], \
                                                 train[-1], test[0], \
                                                 test[-1]))
        
        # Below, we compute the scores related to each training set
        _, train_scores, test_scores = learning_curve( \
        LinearSVC(random_state=0), self._X, self._y, \
        cv=self._yield_indices_for_learning_curve(K), n_jobs=-1, \
        verbose=1000, train_sizes=[1.0])
        
        # We get the indices of the various training sets    
        train_indices, _ = \
        self._generate_list_indices_for_learning_curve(K)
        
        # Below, the compute the values needed to plot the learning curves
        (train_sizes, train_scores_mean, train_scores_std, \
            test_scores_mean, test_scores_std) = \
        self._compute_mean_std_for_train_test_sets(train_indices, \
            train_scores, test_scores)

        # Then, we plot the aforementioned learning curves
        title = "Learning Curves (Linear SVM without tuning, " + \
        self._type + \
        " approach, {} folds)".format(K)
        fig = self._plot_learning_curve(title, "accuracy", \
                                        train_sizes, \
                                        train_scores_mean, \
                                        train_scores_std, \
                                        test_scores_mean, \
                                        test_scores_std)
        
        name_file = "{}_learning_curves_{}_folds.png".format( \
        self._type, K)
        save_file = None if not save_file \
        else os.path.join(self._current_dir, name_file)  

        if save_file:
            plt.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        
        print_log("--- It has taken {} seconds ---" \
        .format(time.time() - start_time))
          
    def _build_data_set(self):
        np.random.seed(0) # We set the seed
        # First we load the data set
        json_data = load_data_set(self._data_set_file)
        
        developers_dict_data = None
#         TO DO
#         load_developers_mappings(self._developers_dict_file)
        developers_list_data = None
#         TO DO
#         load_distinct_developers_list(self._developers_list_file)
        
        # TO DO
        # Then, we build a data frame using the loaded data set, the 
        # loaded developers mappings, the loaded distinct developers.
        self._df = build_data_frame(json_data, developers_dict_data, \
                                    developers_list_data)
#         self._df = self._df[-1000:]
        
        print_log("Shape of the initial Data Frame") # Debug
        print_log(self._df.shape) # Debug
        print_log(self._df['class'].value_counts(normalize=True))
                
        print_log("We count the occurrence of each term") # Debug
        count_vectorizer = CountVectorizer(lowercase=False, \
        token_pattern=u"(?u)\S+")
        X_counts = count_vectorizer \
        .fit_transform(self._df['text'].values)
        
        print_log("Use of the TF model") # Debug
        tfidf_transformer = TfidfTransformer(use_idf=False, \
        smooth_idf=False)
        print_log(X_counts.shape) # Debug
        
        print_log("Computation of the weights of the TF model")
        self._X = tfidf_transformer.fit_transform(X_counts)
        self._y = self._df['class'].values