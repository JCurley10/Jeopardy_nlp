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
import re


def read_tsv(filepath):
    """Reads in a tsv file

    Args:
        filepath (string): filepath and file name of the 
            tsv file to be read into as a pandas dataframe
    Returns:
        Pandas DataFrame
    """    
    return pd.read_csv(filepath, sep = "\t")

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

def lowercase(df, cols):
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
    for col in cols: 
        df[col] = df[col].str.lower()
    return df


def remove_punc(df, cols):
    """[summary]

    Args:
        df ([type]): [description]
        cols (list of str): list of columns to be cleaned, as strings

    Returns:
        Pandas DataFrame: with removed punctuation
    """    
    for col in cols:
        p = re.compile(r'[^\w\s]+')
        df[col] = [p.sub(' ', x) for x in df[col].tolist()]
    return df


def tokenize(df, cols):
    """
    Tokenizes all the words in a stringified column
    Args:
        df (Pandas dataFrame): The dataframe in use
        col (str): the column name to turn into a string
    Returns:
        a string
    """        
    for col in cols:
        df[col] = df[col].apply(word_tokenize)
    return df

def make_stopwords(col):

    if col == 'notes':
        remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    else:
        remove_words = {'man', 'men', 'woman', 'person', 'he', 'she', 'they', 'people', 'with'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    return list(stopwords_set)

#TODO: clean up this function - it's not working properly
def remove_stopwords(df, col):
    """[summary]
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """
    
    df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_set)]))
    return df


# This function doesn't need anything from the above
def clean_text(df, col):
    '''
    """    
    taking in a column from a dataframe,
    return a list of tokenized words, stripped of stopwords 
    Args
            df (Pandas DataFrame)
            cols (list of str): list of columns to be cleaned, as strings
    Returns:
        [list of str]: a list of strings
            to be used for an EDA wordcloud from a given column
    """
    '''
    text = ' '.join(df[col])
    tokens = word_tokenize(text)
    # converts the tokens to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalnum()]
    
    # filter out stop words
    if col == 'notes':
        #TODO: add another set of stopwords for the notes
        remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate', 'night', 'jeopardy', 'night', 'games'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    else:
        stopwords_set = set(stopwords.words('english'))
    words = [w for w in words if not w in stopwords_set]
    return words






def make_q_and_a_col(df):
    """
    Makes a column that concatenates the strings
    from the question and answer columns

    Args:
        df (Pandas DataFrame): 
    Returns:
        Pandas DataFrame with an additional column
    """    
    df['question_and_answer'] = df["question"] + ' ' + df['answer']
    return df

#TODO: make another condition using "view assummptions" 
def make_clue_difficulty_col(df, viewer_assumptions = False):
    if viewer_assumptions:
        conditions = [((df['daily_double']=='no') & (df['value']<= 800)), #easy
                    ((df['daily_double']=='no') & (df['round']== 1) & (df['value'] >= 800)), #average
                    ((df['daily_double']=='no') & (df['value'] == 1200)) #average
                    ((df['daily_double']=='no') & (df['round']== 2) & (df['value'] >= 1600)), #hard
                    ((df['daily_double']== 'yes') & (df['round'] == 1)), #average
                    ((df['daily_double']== 'yes') & (df['round'] == 2)), #hard
                    (df['round'] == 3)] # final jeopardy, hard 

        difficulties = ['easy', 'average', 'average', 'hard', 'average', 'hard', 'hard']
        
    else:
        conditions = [((df['value']<=600) & (df['daily_double']=='no')), #easy
                ((df['daily_double']=='no') & ((df['value']==800) | (df['value']==1200))), #average
                ((df['daily_double']== 'yes') & (df['round'] == 1)), #average
                ((df['daily_double']=='no') & ((df['value']==1000) | (df['value']>=1600))), #hard
                ((df['daily_double']== 'yes') & (df['round'] == 2)), #hard
                (df['round'] == 3)] # final jeopardy, hard 
    


        difficulties = ['easy', 'average', 'average', 'hard', 'hard', 'hard']

    df['clue_difficulty'] = np.select(conditions, difficulties)
    return df

#TODO: write docstring
def update_df_columns(df):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    df_new = make_q_and_a_col(df)
    df_new = make_clue_difficulty_col(df_new)
    return df_new


#TODO: write docstring
def make_sub_df(df, fraction = .05, state = 123):
    """[summary]

    Args:s
        df ([type]): [description]
        fraction (float, optional): [description]. Defaults to .05.
        state (int, optional): [description]. Defaults to 123.

    Returns:
        [type]: [description]
    """
    return df.sample(frac = fraction, axis = 0, random_state = state)


if __name__ == "__main__":
    jeopardy_df = read_tsv('../data/master_season1-35.tsv')
    # jeopardy_df = clean_columns(jeopardy_df, ['category', 'comments', 'answer', 'question'])
    jeopardy_df = update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)
    
    regular_episodes_sub = make_sub_df(regular_episodes)

    no_punc = remove_punc(regular_episodes_sub, ['category', 'question_and_answer'])