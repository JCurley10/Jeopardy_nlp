import nltk
import re
import string
import pandas as pd
import numpy as np
# NKTK
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


def convert_col_to_list(df, col='Question and Answer'):
    """
    takes in a column from a dataframe and
    returns a list of text
    Args:
        df (Pandas DataFrame):
        col (str): column name. Default to "Question and Answer"
    Returns:
        a list of strings, where each column entry is one string
        to be tokenized later
    """
    text = df[col].values.tolist()
    return text


def make_stopwords(filepath='stopwords.txt'):
    """
    read in a list of stopwords from a .txt file
    and extend the nltk stopwords by this list.
    Return a list of stopwords created from nltk
    and the .txt file
    """
    sw = open(filepath, "r")
    my_stopwords = sw.read()
    my_stopwords = my_stopwords.split(", ")
    sw.close()

    all_stopwords = stopwords.words('english')
    all_stopwords.extend(my_stopwords)
    return all_stopwords


def remove_hyphens(text):
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', text)


# tokenize text
def tokenize(text):
    wt = nltk.RegexpTokenizer(pattern=r'\s+', gaps=True)
    tokens = wt.tokenize(text)
    return tokens


def remove_characters(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    no_char_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return no_char_tokens


def lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]


def remove_stopwords(tokens):
    stopword_list = make_stopwords()
    no_stop_tokens = [token for token in tokens if token not in stopword_list]
    return no_stop_tokens


def lemmatized_words(tokens):
    lemmas = []
    for word in tokens:
        lemma = wn.morphy(word)
        if lemma:
            lemmas.append(lemma)
        else:
            lemmas.append(word)
    return lemmas


def remove_short_tokens(tokens):
    return [token for token in tokens if len(token) > 3]


def remove_non_wordnet(tokens):
    return [token for token in tokens if wn.synsets(token)]


def apply_lemmatize(tokens, wnl=WordNetLemmatizer()):
    return [wnl.lemmatize(token) for token in tokens]


def clean_text_clues(texts):
    # text = uncleaned text from list of string 
    # get the text output from convert_col_to_list function from preprocessing_1
    # returns a list of strings where each string is a clue
    clean_clues = []
    for clues in texts:
        clue = remove_hyphens(clues)
        clue = tokenize(clue)
        clue = remove_characters(clue)
        clue = lowercase(clue)
        clue = remove_stopwords(clue)
        clue = lemmatized_words(clue)
        clue = remove_short_tokens(clue)
        clue = remove_non_wordnet(clue)
        clue = apply_lemmatize(clue)
        clean_clues.append(clue)
    return [' '.join(item) for item in clean_clues]


if __name__ == "__main__":

    regular_episodes = pd.read_csv('../data/regular_episodes.csv')
    subsample = regular_episodes.sample(frac=0.02)
    text = subsample['Question and Answer'].values.tolist()

    clean_clues_text = clean_text_clues(text)
    # print(clean_clues_text)
