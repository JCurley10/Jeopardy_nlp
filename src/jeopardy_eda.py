import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import string
import preprocessing 


#TODO: fix up what is happening with the stemming of words in the answers df
    # the Jupyter Notebook has both options 

    
def make_word_cloud(df, col, color, save = False, eda = False):

    #generate word list
    word_lst = preprocessing.clean_text(df, col) # a category was eliminated by removing punctuation 
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
        

def get_wrd_cts(df):
    """[summary]

    Args:
        df (Pandas DataFrame): must have the columns "clue_difficulty", "answer", "question"
    """    
    total_wrds_questions = [len(x.split()) for x in df['Question'].tolist()]
    total_wrds_answers = [len(x.split()) for x in df['Answer'].tolist()]

    data = {'Clue Difficulty': list(df['Clue Difficulty']), 
        'Answer Word Count': total_wrds_answers, 
        'Question Word Count': total_wrds_questions}

    df = pd.DataFrame(data, columns = ['Clue Difficulty', 'Answer Word Count', 'Question Word Count'])
    return df


def graph_wrd_cts(df, col, color, save = False):
    """
    makes a bar graph of the average word counts

    Args:
        df (Pandas DataFrame): must be a dataframe with wordcounts
            can be made from get_wrd_cts
    """    
    fig, ax = plt.subplots(1, 1,figsize = (6, 4), dpi = 150)

    rects1 = ax.bar(df.index, df[col], color = color)
    ax.set_title(f'{col} vs. Clue Difficulty')
    ax.set_ylabel(f'{col}', fontsize = 14)
    ax.set_xlabel('Clue Difficulty', fontsize = 14)
    ax.set_title(f'Clue Difficulty vs.  {col}', fontsize = 16)
    plt.ylim(0, 16)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.025*height,
                    '%f' % float(height),
            ha='center', va='bottom', fontweight = 'bold')
            
    autolabel(rects1)
    plt.tight_layout()
    if save:
        plt.savefig(f'../images/eda_images/{col}_counts_bar.png')

def top_categories(df, n):
    """[summary]

    Args:
        df ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """    
    common_topics = df['J-Category'].value_counts()[:n]
    common_topics
    common_cats = pd.DataFrame(common_topics).reset_index().rename(columns = {"index":"J-Category", "J-Category":"Counts"})
    counts = common_cats['Counts'].apply(lambda x: x / 5)
    pd.Series(counts)
    common_cats['Counts'] = pd.Series(counts)
    common_cats = common_cats.set_index('J-Category')
    return common_cats

def graph_top_categories(df, color, save = False):
    """[summary]

    Args:
        df ([type]): [description]
        save (bool, optional): [description]. Defaults to False.
    """    
    fig, ax = plt.subplots(1, 1, figsize = (6, 4), dpi = 140)
    ax.bar(common_cats.index, common_cats['Counts'], color = color)
    ax.set_ylabel("Number of Episodes", fontsize = 14)
    ax.set_title("Top 10 J-Categories", fontsize = 14)
    ax.set_xlabel("J-Categories", fontsize = 16)
    plt.xticks(rotation=70, ha = 'center', fontstretch = 'semi-condensed', fontsize = 8)
    plt.tight_layout()

    if save:
        plt.savefig('../images/eda_images/top_10_categories.png')
    else:
        plt.show()

if __name__ == "__main__":
    regular_episodes = pd.read_csv("../data/jeopardy_regular_episodes.csv")
    regular_episodes_sub = preprocessing.make_sub_df(regular_episodes)

    make_word_cloud(regular_episodes, 'J-Category',  color = 'ocean', save = True ) #handle how removing punctution affected this 
    # make_word_cloud(regular_episodes, 'Question',  color = 'plasma', save = False)
    # make_word_cloud(regular_episodes, 'Answer',  color = 'plasma', save = False)
    make_word_cloud(regular_episodes, 'Question and Answer', color = 'plasma', save = True )

    df = get_wrd_cts(regular_episodes)
    avgs = df.groupby('Clue Difficulty').mean().sort_values('Answer Word Count').round(2)
    maxes = df.groupby('Clue Difficulty').max().sort_values('Answer Word Count').round(2)
    # graph_wrd_cts(avgs, 'Answer Word Count', color = 'purple', save = False)
    # graph_wrd_cts(avgs, 'Question Word Count', color = 'darkorange', save = False)

    # top_categories(jeopardy_df, 10)
    common_cats = top_categories(regular_episodes, 10)
    # print(common_cats)
    graph_top_categories(common_cats, color = "steelblue", save = True)
    

