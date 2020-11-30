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


def get_word_indices(df, col_name):
    words = df[col_name].values
    count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english',
                             analyzer='word', max_df=1.0, min_df=1,
                             max_features=None)
    count_vect.fit(words)
    count_vect.transform(words)
    return count_vect.vocabulary_ 
    
def hand_label_topics(H, vocabulary):
    '''
    Print the most influential words of each latent topic, and prompt the user
    to label each topic. The user should use their humanness to figure out what
    each latent topic is capturing.
    '''
    hand_labels = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:20]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_five]))
    return hand_labels

def get_topics(col_name, num):
    '''
    col_name (str): input the column name we want to get the latent topics of 
    num (int): number of topics we want to get
    '''
    words = df[col_name].values
    vectorizer = TfidfVectorizer(stop_words = 'english')
    X = vectorizer.fit_transform(words)
    vectorizer.vocabulary_
    vocabulary = vectorizer.get_feature_names()
    vocabulary = np.array(vocabulary)


    nmf = NMF_sklearn(n_components=num, max_iter=100, random_state=12345, alpha=0.0)
    # W = nmf.fit_transform(X)
    H = nmf.components_
    print('reconstruction error:', nmf.reconstruction_err_)

    return hand_label_topics(H, vocabulary)


def build_text_vectorizer(text):
    """[summary]
    Args:
        text (string or series): the text that will be fit to the 
                text_vectorizer whose words will be counted 
        use_tfidf (bool, optional): . Defaults to True.
        stop_words (string, optional). Defaults to 'english'
        use_stemmer (bool, optional): [description]. Defaults to False.
        max_features ([type], optional): [description]. Defaults to None.
    Returns:
    the a list of strings that are the words that appear in the text
    """       
    pass 
    count_vect = CountVectorizer(ngram_range = (1, 2), use_tfidf=True, lowercase=True, use_stemmer=False, tokenizer=None, stop_words='english',  max_features=None)
    count_vect.fit_transform(text)
    return count_vect.vocabulary_


if __name__ == "__main__":
    pass