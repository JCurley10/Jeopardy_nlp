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
from cleaning import clean_text
from cleaning import read_tsv
from build_features import update_df_columns


#TODO: fix up what is happening with the stemming of words in the answers df
    # the Jupyter Notebook has both options 

    
def make_word_cloud(df, col, color, save = False):

    #generate word list
    word_lst = clean_text(df, col)
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
        plt.savefig(f'../images/eda_images/{col}_wordcloud.png')
    else:
        plt.show()


if __name__ == "__main__":
    jeopardy_df = read_tsv('../data/master_season1-35.tsv')
    jeopardy_df = update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)


    # make_word_cloud(jeopardy_df, 'category',  color = 'plasma', save = False )
    # make_word_cloud(jeopardy_df, 'question',  color = 'plasma', save = False)
    # make_word_cloud(jeopardy_df, 'answer',  color = 'plasma', save = False)
    make_word_cloud(jeopardy_df, 'question_and_answer', color = 'plasma', save = True )




