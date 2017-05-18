# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.svm import LinearSVC
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import NMF
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utilities import load_data_set, build_data_frame, print_log
# load_developers_mappings, load_distinct_developers_list, 
import time
import numpy as np
import logging

def main():
    logging.basicConfig(filename='classify_k_folds.log', \
                        filemode='w', level=logging.DEBUG)
    data_set_file = "output.json" # The path of the file which 
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

    # Then, we build a data frame using the loaded data set, the 
    # loaded developers mappings, the loaded distinct developers.
    df = build_data_frame(json_data, developers_dict_data, developers_list_data)

#     TO DO: Fix the lines below later 
#     print_log("Histogram of the the frequencies of each class")
#     pd.value_counts(df['class'], sort=False).hist()
#     pylab.show()

    kf = KFold(n_splits=10) # We instantiate a 10-Folds
    # cross-validator
    i = 1 # Used to know on each fold we will test the classifier 
    # currently trained
   
    # Below, there is a dictionary to store the names, the classifiers 
    # used, the parameters sent to the constructor of the classifiers 
    # and the fitted classifiers
    models = { \
        "RDF": [RandomForestClassifier, {
            "n_estimators": 10,
            "n_jobs": -1
        }, None], \
        "NB": [MultinomialNB, {}, None], \
#         "SVM": [SVC, {}, None], \
        "Linear SVM": [LinearSVC, {}, None], \
        "Logistic Regression": [LogisticRegression, {
            "n_jobs": -1
        }, None], \
        "Stochastic Gradient Descent": [SGDClassifier, {
            "n_jobs": -1
        }, None], \
        "Nearest Centroid": [NearestCentroid, {}, None] \
#         "K Nearest Neighbors": [KNeighborsClassifier, { \
#            n_jobs=-1
#         }, None]
    }
    # Below, there is a dictionary of lists to store the 
    # accuracies related to each fold (for each classifier)
    models_predictions = {}

    for train_index, test_index in kf.split(df):       
        print_log("********* Evaluation on fold {} *********"\
        .format(i)) # Debug

        print_log("We count the occurrence of each term") # Debug
        count_vectorizer = CountVectorizer(max_df=0.5,\
            max_features=50000)
        X_train_counts = count_vectorizer \
        .fit_transform(df.iloc[train_index]['text'].values)
        print_log("Use of the TF-IDF model") # Debug
        tfidf_transformer = TfidfTransformer() 
        print_log(X_train_counts.shape)
        # Debug
        print_log("Computation of the weights of the TF-IDF model")
        X_train = tfidf_transformer.fit_transform(X_train_counts)
        y_train = df.iloc[train_index]['class'].values
        print_log(X_train.shape)
        
        print_log("Training of the models") # Debug
        for key, value in models.items():
            print_log(key)
            models[key][-1] = models[key][0](**models[key][1]) \
            .fit(X_train, y_train)

        print_log("We count the occurrence of each " + \
            "term in the val. set") # Debug
        X_val_counts = count_vectorizer \
        .transform(df.iloc[test_index]['text'].values)
        print_log("Computation of the weights of the TF-IDF " + \
        "model for the validation set") # Debug
        X_val = tfidf_transformer.transform(X_val_counts)
        y_val = df.iloc[test_index]['class'].values
        print_log("Making predictions") # Debug
        
        for key, value in models.items():
            if i == 1:
                models_predictions[key] = []
            models_predictions[key].append(\
            np.mean(value[-1].predict(X_val) == y_val))

        i += 1

    # Below, we print the average accuracies
    for key, value in models_predictions.items():
        print_log("{} Accuracy".format(key)) # Debug
        print_log(sum(value)/len(value)) # Debug

    print_log("--- {} seconds ---".format(time.time() - start_time))

    # We dump the data frame
    df.to_csv("pre_processed_data.csv")

if __name__ == "__main__":
    main()