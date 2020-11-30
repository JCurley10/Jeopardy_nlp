import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn import svm
from build_features import update_df_columns
from cleaning import read_tsv


def make_train_test_sets(df, x_cols, y_col, test_size = .25, random_state = 123):
    """[summary]

    Args:
        df ([type]): [description]
        x_cols (string or list of strings): the columns to use as training features
        y_col (string): the column to use as the target 
        test_size (float, optional): size of the test set. Defaults to .25.
        random_state (int, optional): random state. Defaults to 123.

    Returns:
        tuple: X_train, X_test, y_train, y_test 
            X_train and X_test are either
            Pandas DataFrame if >1 columns are passed as features, 
            or Series object if 1 column is passed
            y_train, y_test are Series objects
            
    """    
    X = df[x_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


def build_text_vectorizer(X_train):
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
    count_vect = CountVectorizer()
    x_train_vectors = count_vect.fit_transform(X_train)
    x_test_vectors = count_vect.transform(X_test)
    return x_train_vectors


if __name__ == "__main__":
    jeopardy = read_tsv('../data/master_season1-35.tsv')
    jeopardy_df = update_df_columns(jeopardy)

    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournament = jeopardy_df.drop(regular_episodes.index)

    x_cols = 'question_and_answer'
    y_col = 'question_difficulty'
    X_train, X_test, y_train, y_test = make_train_test_sets(regular_episodes, x_cols, y_col, test_size = .25, random_state = 123)
    X_train_sample = X_train.sample(frac = .1, axis = 0, random_state = 123)

    print (build_text_vectorizer(X_train_sample).toarray())
  
    