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
s

def get_sub_df(df, testsize = .2, sub_fraction = 0.01, state = 123):
    """
    split the dataframe into training and testing sets
    and get a smaller subset of the training dataframe

    Args:
        df (Pandas dataFrame): The dataframe in use
        training_fraction (float, optional): the fraction of the dataframe 
            to be included in the training set. Defaults to 0.8.
        sub_fraction (float, optional): the fraction of the dataframe
            to be included in the subset to run methods and function on.
             Defaults to 0.01.
        state (int, optional): random state to use. Defaults to 123.

    Returns:
        tuple of dataframes: training: training dataframe
                            testing: testing dataframe
                            sub_training: subset of the training dataframe
    """    
    training, testing = train_test_split(df, test_size = testsize, random_state = state)
    sub_training = training.sample(frac = sub_fraction, axis = 0, random_state = state)
    return training, testing, sub_training



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

def clean_columns(df, col, stem = None):
    """
        cleans the columns by converting to a string, 
    lowercasing, removing stopwords, and tokenizing all in one 

    Args:
        df (pandas DataFrame): the DataFrame whose columns 
                will be turned to one string
        col (string): The column name in question
        stem (string), (optional):
                "snowball" for snowball stemmer
                "porter" for porter stemmer. 
                Defaults to None.

    Returns:
        list: a list of words as strings
    """
    text = stringify(df, col)
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # lemmatize the words 
    wordnet = WordNetLemmatizer()
    lemmatized = [wordnet.lemmatize(word) for word in stripped]
    
    #stem words, depending on the stemmer chosen as a parameter
    if stem == 'snowball':
        snowball = SnowballStemmer('english')
        stemmed = [snowball.stem(word) for word in lemmatized]
        # remove remaining tokens that are not alphabetic from stemmed
        words = [word for word in stemmed if word.isalpha()]
    elif stem == 'porter':
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in lemmatized]
        # remove remaining tokens that are not alphabetic from stemmed
        words = [word for word in stemmed if word.isalpha()]
    elif stem == None:
        pass
        #skip stemming
        # remove remaining tokens that are not alphabetic from lemmatized
    else:
        words = [word for word in lemmatized if word.isalpha()]
    
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