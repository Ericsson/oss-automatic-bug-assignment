# -*- coding: utf-8 -*-
"""
.. module:: classify
   :platform: Unix, Windows
   :synopsis: This module is used to conduct some minor random 
              experiments on any data set used in the thesis.

.. moduleauthor:: Daniel Artchounin <daniel.artchounin@ericsson.com>


"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import VotingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
# from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
from sklearn.svm import LinearSVC
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import NMF
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from mlxtend.classifier import StackingClassifier
# from mlxtend.classifier import SoftmaxRegression
# from mlxtend.classifier import MultiLayerPerceptron
# from mlxtend.preprocessing import DenseTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import LabelBinarizer
from utilities import load_data_set, build_data_frame
# load_developers_mappings, load_distinct_developers_list,
from wordcloud import WordCloud
from utilities import print_log
# from os import path
import seaborn as sn
import matplotlib.pyplot as plt
# import matplotlib
# import pylab
import time
import numpy as np
import pandas as pd
# import scipy.sparse as sp
import os
import inspect
import logging

def main():
#     matplotlib.style.use('ggplot')
    logging.basicConfig(filename='classify.log', filemode='w', \
    level=logging.DEBUG)
    current_dir = os.path.dirname(os.path.abspath( \
    inspect.getfile(inspect.currentframe())))

    data_set_file = "./pre_processing_experiments/output_with_cleaning_without_stemming_without_lemmatizing_with_stop_words_removal_with_punctuation_removal_with_numbers_removal.json" # The path of the file which 
    # contains the pre-processed output
    # Below, the path of the file which contains a dictionary related 
    # to the mappings of the developers
    developers_dict_file = "../developers_dict.json"
    # Below, the path of the file which contains a list of the 
    # relevant distinct developers
    developers_list_file = "../developers_list.json"

    np.random.seed(0) # We set the seed

    start_time = time.time() # We get the time expressed in seconds 
    # since the epoch

    # First we load the data of the three aforementioned files
    json_data = load_data_set(data_set_file)
    developers_dict_data = load_developers_mappings(developers_dict_file)
    developers_list_data = load_distinct_developers_list(developers_list_file)

#     sm = SMOTE(random_state=42)

    # Then, we build a data frame using the loaded data set, the 
    # loaded developers mappings, the loaded distinct developers.
    df = build_data_frame(json_data, developers_dict_data, developers_list_data)
        
    s = " ".join([tr.lower() for tr in df['text'].tolist()])
        
    wordcloud = WordCloud(max_font_size=40).generate(s)
    fig = plt.figure()
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud)
    plt.axis("off")
    
    save_wordcloud_file = os.path.join(current_dir, "wordcloud.png")
    
    if save_wordcloud_file:
        plt.savefig(save_wordcloud_file, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()    
        
#     TO DO: Fix the lines below later 
#     print("Histogram of the the frequencies of each class")
#     pd.value_counts(df['class'], sort=False).hist()
#     pylab.show()
    
    # Below, there is a dictionary to store the names, the classifiers 
    # used, the parameters sent to the constructor of the classifiers 
    # and the fitted classifiers
    models = { \
        "RDF": [RandomForestClassifier, {
            "n_estimators": 50,
            "n_jobs": -1
        }, None], \
#         "ExtraTreesClassifier": [ExtraTreesClassifier, {
#             "n_estimators": 50,
#             "n_jobs": -1
#         }, None], \
        "NB": [MultinomialNB, {}, None], \
#         "SVM": [SVC, { \
#             "kernel": "linear", \
#             "probability": True
#         }, None], \
        "Perceptron": [Perceptron, { \
           "n_jobs": -1, \
           "class_weight": "balanced"
        }, None], \
        "PassiveAggressiveClassifier": [PassiveAggressiveClassifier, { \
            "n_jobs": -1, \
            "class_weight": "balanced"
        }, None], \
#         "RidgeClassifier": [RidgeClassifier, { \
#             "solver": "sag", \
#             "normalize": True
#         }, None], \
        "RidgeClassifier (with wrapper)": [OneVsRestClassifier, { \
            "n_jobs": -1, \
            "estimator": RidgeClassifier(solver="sag", \
            normalize=True \
#             class_weight="balanced" # Bug: should be fixed
            )
        }, None], \
        "Linear SVM": [LinearSVC, { \
            "random_state": 0, \
            "class_weight": "balanced" \
        }, None], \
        "Linear SVM (with wrapper)": [OneVsRestClassifier, { \
            "n_jobs": -1, \
            "estimator": LinearSVC(random_state=0, class_weight="balanced") \
        }, None], \
        "CalibratedClassifierCV (Linear SVM with wrapper)": [CalibratedClassifierCV, { \
            "base_estimator": OneVsRestClassifier(n_jobs=-1, \
            estimator=LinearSVC(random_state=0, class_weight="balanced")) \
        }, None], \
        "Logistic Regression": [LogisticRegression, {
            "n_jobs": -1, \
            "class_weight": "balanced" \
#             "multi_class": "multinomial", \
#             "solver": "newton-cg"
        }, None], \
        "Stochastic Gradient Descent": [SGDClassifier, {
            "n_jobs": -1, \
            "n_iter": 50, \
            "shuffle": True, \
            "class_weight": "balanced"
        }, None], \
        "Nearest Centroid": [NearestCentroid, {}, None],        
#         "RadiusNeighborsClassifier": [RadiusNeighborsClassifier, { \
#         }, None]
#         "LinearDiscriminantAnalysis": [LinearDiscriminantAnalysis, \
#         { \
#         }, None], \
#         "QuadraticDiscriminantAnalysis": [ \
#         QuadraticDiscriminantAnalysis, { \
#         }, None], \
#         "K Nearest Neighbors": [KNeighborsClassifier, { \
#         }, None], \
#         "DecisionTreeClassifier": [DecisionTreeClassifier, { \
#         }, None], \
#         "Bagging Linear SVM": [BaggingClassifier, { \
#             "base_estimator": LinearSVC(random_state=0, \
#         class_weight="balanced"), \
#             "max_samples": 0.5, \
#             "max_features": 0.5, \
#             "random_state": 0, \
#             "n_jobs": -1, \
#             "n_estimators": 100
#         }, None], \
#         "Bagging K Nearest Neighbors": [BaggingClassifier, { \
#             "base_estimator": KNeighborsClassifier() \
#         }, None]
#         "AdaBoostClassifier": [AdaBoostClassifier, { \
#             "base_estimator": SGDClassifier(loss="log", n_jobs=-1, \
#             n_iter=50, shuffle=True, class_weight="balanced"), \
#             "n_estimators": 50 \
#         }, None], \
#         "GradientBoostingClassifier": [GradientBoostingClassifier, \
#         { \
#             "n_estimators": 10 \
#         }, None] \
#         "VotingClassifier": [VotingClassifier, { \
#             "estimators": [ \
#                 ("pac", PassiveAggressiveClassifier(n_jobs=-1)),
#                 ("rc", RidgeClassifier(solver="sag")),
#                 ("lsvc", LinearSVC()),
#                 ("lr", LogisticRegression(n_jobs=-1)),
#                 ("sgdc", SGDClassifier(n_jobs=-1, n_iter=50))
#             ], \
#             "voting": "hard", \
#             "n_jobs": -1
#         }, None],
#         "VotingClassifier2": [VotingClassifier, { \
#             "estimators": [ \
#                 ("lr", LogisticRegression(n_jobs=-1)), \
#                 ("sgdc", SGDClassifier(loss="modified_huber", \
#                 n_jobs=-1, n_iter=50)), \
#                 ("RandomForestClassifier", RandomForestClassifier( \
#                 n_estimators=50, n_jobs=-1)) \
#             ], \
#             "voting": "soft", \
#             "n_jobs": -1
#         }, None],
#         "StackingClassifier": [StackingClassifier, { \
#             "classifiers": [ \
#                 SGDClassifier(loss="modified_huber", \
#                 n_jobs=-1, n_iter=50), \
#                 CalibratedClassifierCV(base_estimator= \
#                 OneVsRestClassifier(n_jobs=-1, \
#                 estimator=LinearSVC(random_state=0))), \
#                 CalibratedClassifierCV(base_estimator= \
#                 OneVsRestClassifier(n_jobs=-1, \
#                 estimator=RidgeClassifier(solver="sag"))) \
#             ], \
#             "use_probas": True, \
#             "average_probas": False, \
#             "meta_classifier": LogisticRegression(n_jobs=-1)
#         }, None]
#         "SoftmaxRegression": [SoftmaxRegression, {}, None], \
#         "MultiLayerPerceptron": [MultiLayerPerceptron, {}, None]    
    }
    
    # Below, there is a dictionary to store the accuracies for each 
    # classifier
    models_predictions = {}

    chi2_feature_selection = SelectKBest(chi2, k="all")
    
    print_log("Splitting the data set") # Debug
#     df = df[-30000:]
    train_set, val_set, test_set = np.split(df, \
        [int(.9*len(df)), int(.9999999999999999999999*len(df))])

#     train_set, val_set, test_set = np \
#     .split(df.sample(frac=1), \
#            [int(.6*len(df)), int(.8*len(df))])

    print_log("Shape of the initial Data Frame") # Debug
    print_log(df.shape) # Debug
    print_log(df['class'].value_counts(normalize=True))
    print_log("Shape of the training set") # Debug
    print_log(train_set.shape) # Debug
    print_log(train_set['class'].value_counts(normalize=True))
    print_log("Shape of the validation set") # Debug
    print_log(val_set.shape) # Debug
    print_log(val_set['class'].value_counts(normalize=True))
    print_log("Shape of the test set") # Debug
    print_log(test_set.shape) # Debug
    print_log(test_set['class'].value_counts(normalize=True))
    print_log("We count the occurrence of each term") # Debug
    
    count_vectorizer = CountVectorizer( \
        lowercase=False, \
        token_pattern=u"(?u)\S+"
#        ngram_range=(1,2) \
#        max_df=0.5, \
#        max_features=100000
    )
    print_log("Size of the vocabulary")
    X_train_counts = count_vectorizer \
    .fit_transform(df['text'].values)
    print_log(X_train_counts.shape)
    X_train_counts = count_vectorizer \
    .fit_transform(train_set['text'].values)
    print_log(X_train_counts.shape)
    print_log("Use of the TF-IDF model") # Debug
    tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=False) 
    # Debug
    print_log("Computation of the weights of the TF-IDF model")
    X_train = tfidf_transformer.fit_transform(X_train_counts)
    y_train = train_set['class'].values

#     standard_scaler = StandardScaler(with_mean=False)
#     
#     X_train = standard_scaler.fit_transform(X_train, y_train)
# 
#     print("Shape of the training set before over sampling") # Debug
#     print(X_train.shape) # Debug
# 
#     X_train, y_train = resample(X_train, y_train, random_state=0)

    X_train = chi2_feature_selection.fit_transform(X_train, y_train)
#     X_train_dense = None
#     y_train_dense = None
# 
#     dense_transformer = DenseTransformer()
#     le = LabelEncoder()

#     X_train, y_train = sm.fit_sample(X_train.toarray(), y_train)
# 
#     if hasattr(X_train, 'dtype') and np.issubdtype(X_train.dtype, np.float):
#         # preserve float family dtype
#         X_train = sp.csr_matrix(X_train)
#     else:
#         # convert counts or binary occurrences to floats
#         X_train = sp.csr_matrix(X_train, dtype=np.float64)
# 
#     print("Shape of the training set after over sampling") # Debug
#     print(X_train.shape)
#     print(pd.Series(y_train).value_counts(normalize=True))

    
    print_log("Training of the models") # Debug
    for key, value in models.items():
        print_log(key)        
#         if key == "SoftmaxRegression" or key == "MultiLayerPerceptron":
#             X_train_dense = dense_transformer.fit_transform(X_train)
#             y_train_dense = le.fit_transform(y_train)
#             models[key][-1] = models[key][0](minibatches=1) \
#             .fit(X_train_dense, y_train_dense)
#         else:
#         if key == "LinearDiscriminantAnalysis":
#             models[key][-1] = models[key][0](**models[key][1]) \
#             .fit(X_train.toarray(), y_train)    
#         else:        
        models[key][-1] = models[key][0](**models[key][1]) \
        .fit(X_train, y_train)         
        print_log("--- {} seconds ---".format(time.time() - start_time))
        
    print_log("We count the occurrence of each term in the val. " + \
              "set") # Debug
    X_val_counts = count_vectorizer \
    .transform(val_set['text'].values)
    print_log("Computation of the weights of the TF-IDF model " + \
              "for the validation set") # Debug
    X_val = tfidf_transformer.transform(X_val_counts)
#     X_val = standard_scaler.transform(X_val)
    X_val = chi2_feature_selection.transform(X_val)
    y_val = val_set['class'].values
#     X_val_dense = None
#     y_val_dense = None
    print_log("Making predictions") # Debug
    
    for key, value in models.items():
        print_log(key)
#         if key == "SoftmaxRegression" or key == "MultiLayerPerceptron":
#             X_val_dense = dense_transformer.transform(X_val)
#             y_val_dense = le.transform(y_val)
#             models_predictions[key] = np.mean(value[-1] \
#             .predict(X_val_dense) == y_val_dense)
#         else:
#         if key == "LinearDiscriminantAnalysis":
#             models_predictions[key] = np.mean(value[-1] \
#             .predict(X_val.toarray()) == y_val)
#         else:
        models[key].append(value[-1] \
        .predict(X_val))
        models_predictions[key] = np.mean(models[key][-1] == y_val)
        print_log("--- {} seconds ---".format(time.time() - start_time))

    # Below, we print the accuracy of each classifier
    for key, value in models_predictions.items():
        print_log("Accuracy of {}".format(key)) # Debug
        print_log(value) # Debug
        print_log("Predicted labels")
        print_log(models[key][-1])
        print_log("True labels")
        print_log(y_val)
        try:
            if callable(getattr(models[key][-2], "predict_proba")):
                
#                 print_log(models[key][-2].classes_)
#                 print_log(models[key][-2].predict_proba(X_val))
                
                lb = LabelBinarizer()                
                
                _ = lb.fit_transform(models[key][-2].classes_)
#                 print_log(lb.classes_)
#                 print_log(y_classes_bin)
#                 print_log(lb.transform(["exclude"]))  
              
                y_val_bin = lb.transform(y_val)
                print_log("Mean Reciprocal Rank:")
                print_log(label_ranking_average_precision_score( \
                y_val_bin, models[key][-2].predict_proba(X_val)))
        except AttributeError:
            pass
        print_log("Detailed report:")
        print_log(classification_report(y_val, models[key][-1]))

    print_log("Confusion matrix of Linear SVM (with wrapper):")
    cm = confusion_matrix(y_val, models["Linear SVM"][-1], labels=df['class'].unique())
    print_log(df['class'].unique())
    print_log(cm)
    df_cm = pd.DataFrame(cm, index=df['class'].unique(), columns=df['class'].unique() )
    fig = plt.figure(figsize=(20.0, 12.5))
    sn.set(font_scale=0.5)
    sn.heatmap(df_cm, annot=True, annot_kws={"size":8})

    save_file = os.path.join(current_dir, "confusion_matrix.png")  

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    print_log("--- {} seconds ---".format(time.time() - start_time))
    # We dump the data frame
    df.to_csv("pre_processed_data.csv")
    
if __name__ == "__main__":
    main()