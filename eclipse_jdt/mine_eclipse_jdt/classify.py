# -*- coding: utf-8

import json
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import pylab

matplotlib.style.use("ggplot")

if __name__ == "__main__":
    data_file = "./pre_processed_output.json" # The file which
    # contains the pre-processed output
    with open(data_file) as json_data:
        # Then, we load the JSON contents
        json_data = json.load(json_data)
        rows = []
        index = []
        # Then, we build a data frame
        print("Loading the pre-processed data") # Debug
        for element in json_data:
            if len(element["bug_id"]) != 0 and \
            len(element["title"]) != 0 and \
            len(element["description"]) != 0 and \
            element["assigned_to"] is not None and \
            len(element["assigned_to"]) != 0:
                rows.append({
                    "text":  " ".join(element["title"]) + \
                    " ".join(element["description"]),
                    "class": element["assigned_to"]
                })
                print(element["bug_id"]) # Debug
                index.append(element["bug_id"])
        # We build a Data Frame with the pre-processed data
        data_frame = DataFrame(rows, index=index)
        print("Shape of the Data Frame") # Debug
        print(data_frame.shape) # Debug
        # print("Data Frame is being filtered") # Debug
        # print("Shape of the filtered Data Frame") # Debug
        # print(data_frame.shape) # Debug
        print("Data Frame first lines") # Debug
        print(data_frame.head()) # Debug
        print("We shuffle the rows of the Data Frame") # Debug
        data_frame = data_frame \
        .reindex(np.random.permutation(data_frame.index))
        print("Shuffled Data Frame first lines") # Debug
        print(data_frame.head()) # Debug
        print("Number of classes in the Data Frame") # Debug
        print(len(set(data_frame["class"].values))) # Debug
        print("Frequency of each class") # Debug
        print(pd.value_counts(data_frame["class"] \
            .values, sort=False))
        print("Splitting the data set") # Debug
        train_set, val_set, test_set = np \
        .split(data_frame.sample(frac=1), \
            [int(.6*len(data_frame)), int(.8*len(data_frame))])
        print("Shape of the initial Data Frame") # Debug
        print(data_frame.shape) # Debug
        print("Shape of the initial training set") # Debug
        print(train_set.shape) # Debug
        print("Shape of the initial validation set") # Debug
        print(val_set.shape) # Debug
        print("Shape of the initial test set") # Debug
        print(test_set.shape) # Debug
        print("We count the occurrences of each term") # Debug
        count_vectorizer = CountVectorizer()
        X_train_counts = count_vectorizer \
        .fit_transform(train_set["text"].values)
        print("Use of the TF-IDF model") # Debug
        tfidf_tranformer = TfidfTransformer()
        # Debug
        print("Computation of the weights of the TF-IDF model")
        # Debug
        X_train = tfidf_tranformer.fit_transform(X_train_counts)
        y_train = train_set["class"].values

        print("Training of the models") # Debug
        rdf_clf = RandomForestClassifier(n_estimators=30) \
        .fit(X_train, y_train)
        print("1") # Debug
        nb_clf = MultinomialNB().fit(X_train, y_train)
        print("2") # Debug
        # svm_clf = SVC().fit(X_train, y_train)
        print("3") # Debug
        lin_svm_clf = LinearSVC().fit(X_train, y_train)
        print("4") # Debug
        # log_reg_clf = LogisticRegression().fit(X_train, y_train)
        print("5") # Debug
        sgd_clf = SGDClassifier().fit(X_train, y_train)
        print("6") # Debug
        nc_clf = NearestCentroid().fit(X_train, y_train)
        print("7") # Debug
        knn_clf = KNeighborsClassifier().fit(X_train, y_train)

        # Debug
        print("We count the occurrence of each term in the val. set")
        X_val_counts = count_vectorizer \
        .transform(val_set["text"].values)
        print("Computation of the weights of the TF-IDF model " + \
            "for the validation set") # Debug
        X_val = tfidf_tranformer.transform(X_val_counts)
        y_val =val_set["class"].values
        print("Making predictions") # Debug
        rdf_predictions = rdf_clf.predict(X_val)
        nb_predictions = nb_clf.predict(X_val)
        # svm_predictions = svm_clf.predict(X_val)
        lin_svm_predictions = lin_svm_clf.predict(X_val)
        # log_reg_predictions = log_reg_clf.predict(X_val)
        sgd_predictions = sgd_clf.predict(X_val)
        nc_predictions = nc_clf.predict(X_val)
        knn_predictions = knn_clf.predict(X_val)

        print("RDF Accuracy") # Debug
        print(np.mean(rdf_predictions == y_val)) # Debug
        print("NB Accuracy") # Debug
        print(np.mean(nb_predictions == y_val)) # Debug
        # print("SVM Accuracy") # Debug
        # print(np.mean(svm_predictions == y_val)) # Debug
        print("Linear SVM Accuracy") # Debug
        print(np.mean(lin_svm_predictions == y_val)) # Debug
        # print("Logistic Regression Accuracy") # Debug
        # print(np.mean(log_reg_predictions == y_val)) # Debug
        print("Stochastic Gradient Descent Accuracy") # Debug
        print(np.mean(sgd_predictions == y_val)) # Debug
        print("Nearest Centroid Accuracy") # Debug
        print(np.mean(nc_predictions == y_val)) # Debug
        print("KNN Accuracy") # Debug
        print(np.mean(knn_predictions == y_val)) # Debug

        # TO DO: fix the following lines
        # We dump the data frame
        # data_frame.to_csv("pre_processed_data.csv")