

import numpy as np
import matplotlib.pyplot as plt
import preprocessing 
from sklearn.decomposition import NMF
# for LDA
from sklearn.feature_extraction.text import CountVectorizer


count_vectorizer = CountVectorizer(min_df=10, max_df=0.95, ngram_range=(1,1), stop_words=stopwords, max_features = n_features)

feature_matrix = count_vectorizer.fit_transform(df[col])