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
from sklearn.decomposition import NMF as NMF_sklearn
import string

class Cleaning(object):
    
    def __init__(self):
        pass

    def stringify(self, df, col):
        """[summary]

        Args:
            df ([type]): [description]
            col ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return ' '.join(df[col])

    def lowercase(self, df, col):
        return ' '.join(df[col]).lower()

    #TODO: make the remove_punc function 
    def remove_punc(self, df, col):
        """[summary]

        Args:
            df ([type]): [description]
            col ([type]): [description]

        Returns:
            [type]: [description]
        """        
        return None

    def tokenize(self, df, col):
        """[summary]

        Args:
            df ([type]): [description]
            col ([type]): [description]

        Returns:
            [type]: [description]
        """        
        text = self.stringify(df, col)
        tokenize = [word_tokenize(content) for content in text]
        return tokenize

    def remove_stopwords(self, df, col):
        """[summary]

        Args:
            df ([type]): [description]
            col ([type]): [description]

        Returns:
            [type]: [description]
        """
        docs = df[col].values
        text = Cleaning.stringify(df, col)
        if col == 'notes':
            #TODO: add another set of stopwords for the notes
            remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate'}
            stopwords_set = (set(stopwords.words('english'))).union(remove_words)
        else:
            stopwords_set = set(stopwords.words('english')) 
        return [[word for word in text if word not in stopwords_set] for word in docs]

    def clean_columns(self, df, col):
        """ 
        cleans the columns by converting to a string, 
        lowercasing, removing stopwords, and tokenizing all in one 

        Args:
            df (pandas DataFrame): the DataFrame whose columns 
                                    will be turned to one string
            col (string): The column name in question

        Returns:
            list: a list of words
        """

        text = Cleaning.stringify(df, col)
        tokens = word_tokenize(text)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out sls
        #s top words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        return words

    def build_text_vectorizer(self, contents, use_tfidf=True, use_stemmer=False, max_features=None):
        """[summary]

        Args:
            contents ([type]): [description]
            use_tfidf (bool, optional): [description]. Defaults to True.
            use_stemmer (bool, optional): [description]. Defaults to False.
            max_features ([type], optional): [description]. Defaults to None.
        """        
    pass


if __name__ == '__main__':
    jeopardy = pd.read_csv('../data/master_season1-35.tsv', sep = "\t")
    train_set  = jeopardy.sample(frac = .8, axis = 0, random_state = 123)
    test_set = jeopardy.drop(train_set.index)
    sub_train = train_set.sample(frac = .1, axis = 0, random_state = 123)

    print (Cleaning.remove_stopwords(sub_train, 'category'))
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())