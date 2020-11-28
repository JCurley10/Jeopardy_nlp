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
from cleaning import get_sub_df, clean_columns

#TODO: fix up what is happening with the stemming of words in the answers df
    # the Jupyter Notebook has both options 
    
def make_word_cloud(df, col, color, stem = None, save = False):
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
    word_lst = clean_columns(df, col, stem = None) # can also get the words from the text vectorizer above
    words = ' '.join(word_lst)
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color =None, mode = 'RGBA', 
                colormap = color,
                collocations=False,
                min_font_size = 10).generate(words) 

    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    if save:
        plt.savefig(f'../images/{col}_wordcloud.png')
    else:
        plt.show()


if __name__ == "__main__":
    jeopardy = pd.read_csv('../data/master_season1-35.tsv', sep = "\t")
    regular_tournament = jeopardy[jeopardy['notes']=='-']
    special_tournament = jeopardy.drop(regular_tournament.index)

    jeopardy_train, jeopardy_test, jeopardy_subtrain = get_sub_df(jeopardy)
    regular_train, regular_test, regular_subtrain = get_sub_df(regular_tournament)
    special_train, special_test, special_subtrain = get_sub_df(special_tournament)

    # make_word_cloud(jeopardy, 'category', stem = None, color = 'plasma', save = True)
    # make_word_cloud(jeopardy, 'question', stem = None, color = 'Blues_r', save = True)
    # make_word_cloud(jeopardy_subtrain, 'answer', stem = None, color = 'cividis_r', save = True)

