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
    word_lst = preprocessing.clean_text(df, col)
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
    total_wrds_questions = [len(x.split()) for x in df['question'].tolist()]
    total_wrds_answers = [len(x.split()) for x in df['answer'].tolist()]

    data = {'Clue Difficulty': list(df['clue_difficulty']), 
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
    ax.set_title(f'Clue Difficulty vs. {col}', fontsize = 16)
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
    common_topics = df['category'].value_counts()[:n]
    common_topics
    common_cats = pd.DataFrame(common_topics).reset_index().rename(columns = {"index":"J-Category", "category":"Counts"})
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
    ax.set_ylabel("Number of Episodes J-Category Occurs", fontsize = 14)
    ax.set_title("Top 10 J-Categories", fontsize = 14)
    ax.set_xlabel("J-Categories", fontsize = 16)
    plt.xticks(rotation=70, ha = 'center', fontstretch = 'semi-condensed', fontsize = 8)
    plt.tight_layout()

    if save:
        plt.savefig('../images/eda_images/top_10_categories.png')
    else:
        plt.show()

if __name__ == "__main__":
    jeopardy_df = preprocessing.read_tsv('../data/master_season1-35.tsv')
    jeopardy_df = preprocessing.lowercase(jeopardy_df, ['category'])
    jeopardy_df = preprocessing.remove_punc(jeopardy_df, ['category', 'question', 'answer'])
    jeopardy_df = preprocessing.update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)
    
    regular_episodes_sub = preprocessing.make_sub_df(regular_episodes)


    # make_word_cloud(jeopardy_df, 'category',  color = 'ocean', save = True )
    # make_word_cloud(jeopardy_df, 'question',  color = 'plasma', save = False)
    # make_word_cloud(jeopardy_df, 'answer',  color = 'plasma', save = False)
    # make_word_cloud(jeopardy_df, 'question_and_answer', color = 'plasma', save = True )

    df = get_wrd_cts(special_tournaments)
    avgs = df.groupby('Clue Difficulty').mean().sort_values('Answer Word Count').round(2)
    maxes = df.groupby('Clue Difficulty').max().sort_values('Answer Word Count').round(2)
    # graph_wrd_cts(avgs, 'Answer Word Count', color = 'indigo', save = True)
    # graph_wrd_cts(avgs, 'Question Word Count', color = 'darkorange', save = False)

    # top_categories(jeopardy_df, 10)
    common_cats = top_categories(jeopardy_df, 10)
    graph_top_categories(common_cats, color = "steelblue", save = True)
    

