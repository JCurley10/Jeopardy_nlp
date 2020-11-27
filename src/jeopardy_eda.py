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
import string
# from cleaning import Cleaning

    
'''
TODO: STUFF TO PUT IN A CLASS
'''

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
    removes punctuation from 
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """        
    return None

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

def remove_stopwords(df, col):
    """[summary]
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """
    #TODO: set doc
    docs = df[col].values
    text = stringify(df, col)
    if col == 'notes':
        #TODO: add another set of stopwords for the notes
        remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    else:
        stopwords_set = set(stopwords.words('english')) 
    return [[word for word in text if word not in stopwords_set] for word in docs]

#TODO: this has some issues because it does something wrong with apostrophes 

def clean_columns(df, col):
    """ 
    cleans the columns by converting to a string, 
    lowercasing, removing stopwords, and tokenizing all in one 
    Args:
        df (pandas DataFrame): the DataFrame whose columns 
                                will be turned to one string
        col (string): The column name in question
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
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    if col == 'notes':
        #TODO: add another set of stopwords for the notes
        remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate', 'night', 'games', 'jeopardy'}
        stopwords_set = (set(stopwords.words('english'))).union(remove_words)
    else:
        stopwords_set = set(stopwords.words('english'))
    words = [w for w in words if not w in stopwords_set]
    return words


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
    # count_vect = CountVectorizer(ngram_range = (1, 2),  use_tfidf=True, lowercase=True, use_stemmer=False, tokenizer=None, stop_words='english',  max_features=None)
    # count_vect.fit_transform(text)
    # return count_vect.vocabulary_.keys()


def make_word_cloud(df, col, color, save = False):
    """makes a word cloud of the words per column

    Args:
        df (pandas DataFrame): the DataFrame whose columns 
                will be turned to one string
        col (string): the string in question 
        color (string): the colormap preset to use
        save (bool, optional): Whether to save the figure or just show
                will save the figure as a .png, False will show the figure
                Defaults to False.
    """    
    #generate word list
    word_lst = clean_columns(df, col) # can also get the words from the text vectorizer above
    words = ' '.join(word_lst)
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', colormap = color,
                min_font_size = 10).generate(words) 

    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    if save:
        plt.savefig(f'../data/{col}_wordcloud.png')
    else:
        plt.show()


if __name__ == "__main__":
    jeopardy = pd.read_csv('../data/master_season1-35.tsv', sep = "\t")
    train_set  = jeopardy.sample(frac = .8, axis = 0, random_state = 123)
    test_set = jeopardy.drop(train_set.index) 
    sub_train = train_set.sample(frac = .01, axis = 0, random_state = 123)
    
    train_reg = train_set[train_set['notes']=='-'] #just the questions from a regular episode
    sub_reg = train_reg.sample(frac = 0.1, axis = 0, random_state = 123) #a subsample from the training set of regular episodes
    train_spec = train_set.drop(train_reg.index) #questions from a special episodes/tournaments 
    sub_train = train_spec.sample(frac = 0.2, axis = 0, random_state = 123) #a subsample from the training set of special espisodes/tournaments

    category_string = stringify(sub_train, 'category')
    category_string = clean_columns(sub_train, 'category')

    # make_word_cloud(sub_train, 'category', 'plasma_r', save = True)
    # make_word_cloud(sub_train, 'question', 'Blues_r', save = True)
    # make_word_cloud(sub_train, 'answer', 'cividis_r', save = True)
