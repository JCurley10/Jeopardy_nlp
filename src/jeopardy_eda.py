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
import preprocessor


# TODO: fix up what is happening with the stemming of words in the answers df

def make_word_cloud(df, col, color='cividis', save=False):
    """
    make a wordcloud and either save or show it

    Args:
        df Pandas DataFrame: The dataframe with the columns whose
            words will be used in the wordcloud
        col (str): column name in df whose words will be
            used in a wordcloud
        color (str): matplotlib color name. Default to 'cividis'
        save (bool, optional): [description]. Defaults to False.
    """

    # Generate word list
    word_lst = preprocessing.clean_text(df, col) # a category was eliminated by removing punctuation
    words = ' '.join(word_lst)
    wordcloud = WordCloud(width=800, height=800,
                          background_color=None, mode='RGBA',
                          colormap=color,
                          collocations=False,
                          min_font_size=10).generate(words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    if save:
        plt.savefig(f'../images/eda_images/{col}_wordcloud.png')
    else:
        plt.show();


def get_wrd_cts(df):
    """
    Make a graph with wordcounts and compare
    those wordcounts to the quesiton difficulty

    Args:
        df (Pandas DataFrame): must have the columns "Clue Difficulty",
                                "Answer", "Question"
    """

    total_wrds_questions = [len(x.split()) for x in df['Question'].tolist()]
    total_wrds_answers = [len(x.split()) for x in df['Answer'].tolist()]

    data = {'Clue Difficulty': list(df['Clue Difficulty']),
            'Answer Word Count': total_wrds_answers,
            'Question Word Count': total_wrds_questions}

    df = pd.DataFrame(data, columns=['Clue Difficulty', 'Answer Word Count',
                                     'Question Word Count'])
    return df


def graph_wrd_cts(df, col, color, save=False):
    """
    makes a bar graph of the average word counts

    Args:
        df (Pandas DataFrame): must be a dataframe with wordcounts
            can be made from get_wrd_cts
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)

    rects1 = ax.bar(df.index, df[col], color=color)
    ax.set_title(f'{col} vs. Clue Difficulty')
    ax.set_ylabel(f'{col}', fontsize=14)
    ax.set_xlabel('Clue Difficulty', fontsize=14)
    ax.set_title(f'Clue Difficulty vs.  {col}', fontsize=16)
    plt.ylim(0, 16)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.025*height,
                    '%f' % float(height),
                    ha='center', va='bottom',
                    fontweight='bold')

    autolabel(rects1)
    plt.tight_layout()
    if save:
        plt.savefig(f'../images/eda_images/{col}_counts_bar.png')


def top_categories(df, n):
    """
    get the top J-Categories of all time
    by summing the total appearances of a category
    and dividing by 5 (since each category appears 5 time unless
    its Final Jeopardy, which isn't super important to
    figure out with this analysis)

    Args:
        df Pandas DataFrame: The dataframe with the categories
        n (int): number of top categories

    Returns:
        Pandas DataFrame: DataFrame with the top categories
        counts of how many episodes they appear in
    """
    common_topics = df['J-Category'].value_counts()[:n]
    common_topics
    common_cats = pd.DataFrame(common_topics).reset_index().rename(
                    columns={"index": "J-Category", "J-Category": "Counts"})
    counts = common_cats['Counts'].apply(lambda x: x / 5)
    pd.Series(counts)
    common_cats['Counts'] = pd.Series(counts)
    common_cats = common_cats.set_index('J-Category')
    return common_cats


def graph_top_categories(df, color, save=False):
    """
    make a bargraph to show the top n categories from the
    top_categories function

    Args:
        Pandas DataFrame: DataFrame with the top categories
        save (bool, optional): whether to save or show the function.
        save defaults to False and shows the bargraph.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
    ax.bar(common_cats.index, common_cats['Counts'], color=color)
    ax.set_ylabel("Number of Episodes", fontsize=14)
    # ax.set_title("Top 10 J-Categories", fontsize=14)
    ax.set_xlabel("J-Categories", fontsize=14)
    ax.spines["right"].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(rotation=60, ha='right', fontstretch='semi-condensed',
               fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig('../images/eda_images/top_10_categories_blue.png')
    else:
        plt.show()


if __name__ == "__main__":
    # Read in the jeopardy_regular_episodes.csv file
    regular_episodes = pd.read_csv("../data/regular_episodes.csv")

    # Make wordclouds of the most common words in "J-Category"
    # and "Question and Answer"
    make_word_cloud(regular_episodes, 'J-Category')
    make_word_cloud(regular_episodes, 'Question and Answer', color='plasma',
                    save=False)

    # Graph a barchart of the top 10 most common categories by name
    common_cats = top_categories(regular_episodes, 10)
    graph_top_categories(common_cats, color="midnightblue", save=False)
