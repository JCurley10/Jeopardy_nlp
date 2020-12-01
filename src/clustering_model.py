import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import string
from cleaning import clean_text, read_tsv
from build_features import update_df_columns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF as NMF_sklearn
from sklearn.cluster import KMeans

#TODO: write docstring
def make_sub_df(df, fraction = .05, state = 123):
    """[summary]

    Args:
        df ([type]): [description]
        fraction (float, optional): [description]. Defaults to .05.
        state (int, optional): [description]. Defaults to 123.

    Returns:
        [type]: [description]
    """
    return df.sample(frac = fraction, axis = 0, random_state = state)


#TODO: hyperparameter tuning with the Count Vectorizer 
#TODO: write docstring
def kmeans_cluster(df, col, n):
    """[summary]

    Args:
        df ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """    
    count_vect = CountVectorizer(ngram_range = (1, 1), 
                            lowercase=True,  tokenizer=None, 
                            stop_words='english', analyzer='word',  
                            max_features=None)

    x = count_vect.fit_transform(df[col])
    # features = count_vect.get_feature_names()
    kmeans = KMeans(n_clusters = 10, random_state = 123).fit(x)
    centroids = kmeans.cluster_centers_
    top_n = np.argsort(centroids)[:, :-n+1:-1]
    names = count_vect.get_feature_names()

    name_arr = np.array(names)
    return f'n = {n}', '\n', name_arr[top_n]




if __name__ == "__main__":
    jeopardy_df = read_tsv('../data/master_season1-35.tsv')
    jeopardy_df = update_df_columns(jeopardy_df)
    jeopardy_df_sub = make_sub_df(jeopardy_df)

    km_clusters = kmeans_cluster(jeopardy_df_sub, 'category', 10)
