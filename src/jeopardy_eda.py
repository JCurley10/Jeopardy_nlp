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
    """[summary]
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """        
    return ' '.join(df[col])

def lowercase(df, col):
    return ' '.join(df[col]).lower()

#TODO: make the remove_punc function 
def remove_punc(df, col):
    """[summary]
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """        
    return None

def tokenize(df, col):
    """[summary]
    Args:
        df ([type]): [description]
        col ([type]): [description]
    Returns:
        [type]: [description]
    """        
    text = stringify(df, col)
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

def clean_columns(df, col):
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
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):
    """[summary]
    Args:
        contents ([type]): [description]
        use_tfidf (bool, optional): [description]. Defaults to True.
        use_stemmer (bool, optional): [description]. Defaults to False.
        max_features ([type], optional): [description]. Defaults to None.
    """        
    pass

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
    word_lst = clean_columns(df, col)
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

    category_string = stringify(sub_train, 'category')
    category_string = clean_columns(sub_train, 'category')
    sub_train['category'].values
    # make_word_cloud(sub_train, 'category', 'plasma_r', save = True)
    # make_word_cloud(sub_train, 'question', 'Blues_r', save = True)
    # make_word_cloud(sub_train, 'answer', 'cividis_r', save = True)