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

def stringify(df, col):
    """
    Sets all columns of a dataframe to one string

    Args:
        df (pandas DataFrame): the DataFrame whose columns 
                                will be turned to one string
        col (string): The column name in question

    Returns:
        string : a string whose elements are words from all
                columns
    """    
    return ' '.join(df[col])


def clean_columns(df, col):
    """ cleans the columns by stripping 

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
        plt.savefig(f'{col}_wordcloud.png')
    else:
        plt.show()
    

if __name__ == "__main__":

    jeopardy = pd.read_csv('master_season1-35.tsv', sep = "\t")
    train_set  = jeopardy.sample(frac = .8, axis = 0, random_state = 123)
    test_set = jeopardy.drop(train_set.index)
    sub_train = train_set.sample(frac = .1, axis = 0, random_state = 123)


    make_word_cloud(sub_train, 'category', 'plasma_r', save = True)
    make_word_cloud(sub_train, 'question', 'Blues_r', save = True)
    make_word_cloud(sub_train, 'answer', 'cividis_r', save = True)
