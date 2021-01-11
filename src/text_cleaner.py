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
    Read in a list of stopwords from a .txt file
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
    """
    Remove hyphens from a text as a list of strings
    """
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', text)


# tokenize text
def tokenize(text):
    """
    Tokenize a list of strings
    """
    wt = nltk.RegexpTokenizer(pattern=r'\s+', gaps=True)
    tokens = wt.tokenize(text)
    return tokens


def remove_characters(tokens):
    """
    Remove characters from a list of tokenized text
    """
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    no_char_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return no_char_tokens


def lowercase(tokens):
    """
    Lowercase all text from tokenized text
    """
    return [token.lower() for token in tokens if token.isalpha()]


def remove_stopwords(tokens):
    """
    Remove any stopwords from tokenized text
    """
    stopword_list = make_stopwords()
    no_stop_tokens = [token for token in tokens if token not in stopword_list]
    return no_stop_tokens


def lemmatized_words(tokens):
    """
    Lemmatize any words from tokenized text
    """
    lemmas = []
    for word in tokens:
        lemma = wn.morphy(word)
        if lemma:
            lemmas.append(lemma)
        else:
            lemmas.append(word)
    return lemmas


def remove_short_tokens(tokens):
    """
    Remove tokens that are smaller than 3 characters long
    """
    return [token for token in tokens if len(token) > 3]


def remove_non_wordnet(tokens):
    """
    Remove any tokens that are a part of the synsets library
    """
    return [token for token in tokens if wn.synsets(token)]


def apply_lemmatize(tokens, wnl=WordNetLemmatizer()):
    """
    Lemmatize tokens using nltk's WordNetLemmatizer()
    """
    return [wnl.lemmatize(token) for token in tokens]


def token_by_lemma(text):
    """
    Tokenize and lemmatize text in one function
    Potentially a tokenizer to call as a parameter in NMF model
    -------
    Args:
        text (string): a string to be tokenized
    Returns:
        a list of strings: tokenized words that have been lemmatized
    """
    lemmatizer = WordNetLemmatizer()
    word_list = word_tokenize(text)

    lemmatized_wrds = [lemmatizer.lemmatize(w) for w in word_list]
    return lemmatized_wrds


def clean_text_clues(texts):
    """
    Read in a text as a list of strings and clean the text
    according to the chosen cleaning functions
    -------
    Args:
        texts (list of strings): each string has multiple words separated by 
            whitespace or hyphens which need to be processed and cleand
    Returns:
        list of strings of tokenized and processed text as cleaned_words
    """

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
