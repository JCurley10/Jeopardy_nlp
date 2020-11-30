import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import string
from cleaning import clean_columns
from build_features import * 


#TODO: deal with this text vectorizer and the goal of it. 
    # right now, this does something similar to the clean_columns function above

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