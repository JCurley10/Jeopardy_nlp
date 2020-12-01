import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import string
from cleaning import *
from build_features import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF as NMF_sklearn
from sklearn.cluster import KMeans


#TODO: hyperparameter tuning with the Count Vectorizer 
def kmeans_cluster(df):
    count_vect = CountVectorizer(ngram_range = (1, 2), 
                            lowercase=True,  tokenizer=None, 
                            stop_words='english', analyzer='word',  
                            max_features=None)

    x = count_vect.fit_transform(df)
    features = count_vect.get_feature_names()
    kmeans = KMeans(n_clusters = 10, random_state = 123).fit(x)
    centroids = kmeans.cluster_centers_
    top_10 = np.argsort(centroids)[:, :-11:-1]
    names = count_vect.get_feature_names()

    name_arr = np.array(names)
    return name_arr[top_10]






if __name__ == "__main__":
    pass