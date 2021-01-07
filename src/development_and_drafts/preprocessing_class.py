'''https://www.thinkdatascience.com/post/preprocess-your-text-for-nlp-models-cleaner/'''

import re
import nltk
import spacy
import string
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# from pycontractions import Contractions ## Need to import contractions 

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


class Cleaner():

    def __init__(self, df, col):
        self.df = df
        self.col = col

    def listify(self):
        self.df[self.col].values.tolist()
        return self

    def remove_punc(self):
        p = re.compile(r'[^\w\s]+')
        self.df[self.col] = [p.sub('', x) for x in df[col].tolist()]
        return self

    def lowercase(self):
        self.df[self.col].str.lower()
        return self

    def remove_stopwords(self):
        # make the stopwords
        sw = open("stopwords.txt", "r")
        my_stopwords = sw.read()
        my_stopwords = my_stopwords.split(", ")
        sw.close()
        nltk_stopwords = stopwords.words('english')
        nltk_stopwords.extend(my_stopwords)
        stopwords_set = set(nltk_stopwords).union(my_stopwords)
        # Remove the stopwords
        self.df[self.col] = self.df[self.col].apply(lambda x: ' '.join([word for word in x.split()if word not in (stopwords_set)]))
        return self

    def do_all_cleaning(self):


if __name__ == "__main__":
    df = pd.read_csv("../data/jeopardy_regular_episodes.csv")
    col = 'Question and Answer'
    text = df[col].values.tolist()
    cleaner = Cleaner(df, col)
    cleaner.remove_punc()
    cleaner.lowercase()
    df = cleaner.df
