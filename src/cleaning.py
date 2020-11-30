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
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF as NMF_sklearn
import string


def stringify(df, col):
    """
    Turns the column (col) into one string, 
    to be able to make a wordcloud 

    Args:
        df (Pandas dataFrame): The dataframe in use
        col (str): the column name to turn into a string

    Returns:
        a string
    """        
    return ' '.join(df[col])

def lowercase(df, col):
    """
    turns the column (col) into one string, 
    whose letters are all lowercase
    to be able to make a wordcloud

    Args:
        df (Pandas dataFrame): The dataframe in use
        col (str): the column name to turn into a string

    Returns:
        a string of lowercase letters 
    """    
    return ' '.join(df[col]).lower()

#TODO: make the remove_punc function 
def remove_punc(df, col):
    """
    removes punctuation from a column of text
    Args:
        df (Pandas dataFrame): The dataframe in use
        col (str): the column name to turn into a string
    Returns:
        [type]: [description]
    """
    pass         

def tokenize(df, col):
    """
    Tokenizes all the words in a stringified column
    Args:
        df (Pandas dataFrame): The dataframe in use
        col (str): the column name to turn into a string
    Returns:
        a string
    """        
    text = df[col]
    tokenize = [word_tokenize(content) for content in text]
    return tokenize

#TODO: clean up this function
def remove_stopwords(df, col):
    """[summary]
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """
    docs = df[col].values
    text = stringify(df, col)
    if col == 'notes':
        #TODO: add more to set of stopwords for the notes
        remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    else:
        stopwords_set = set(stopwords.words('english')) 
    return [[word for word in text if word not in stopwords_set] for word in docs]


##TODO: fix up the stem and lemmatizer since they seem to be mucking up the words
def clean_columns(df, col):
    '''
    using a pre-made function 
    returns a list of the tokenized and stripped of stopwords 
    '''
    text = ' '.join(df[col])
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]
    
    # filter out stop words
    if col == 'notes':
        #TODO: add another set of stopwords for the notes
        remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate', 'night', 'jeopardy', 'night', 'games'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    else:
        stopwords_set = set(stopwords.words('english'))
    words = [w for w in words if not w in stopwords_set]
    return words




if __name__ == "__main__":
    jeopardy = pd.read_csv('../data/master_season1-35.tsv', sep = "\t")
    regular_tournament = jeopardy[jeopardy['notes']=='-']
    special_tournament = jeopardy.drop(regular_tournament.index)

    jeopardy_train, jeopardy_test, jeopardy_subtrain = get_sub_df(jeopardy)
    regular_train, regular_test, regular_subtrain = get_sub_df(regular_tournament)
    special_train, special_test, special_subtrain = get_sub_df(special_tournament)

    clean_category = clean_columns(regular_subtrain, 'category')
    clean_category_str = ' '.join(clean_category)