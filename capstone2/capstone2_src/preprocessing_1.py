    """[summary]
    THIS FILE IS A CLEANER FOR TEXT. IT IS ONE OF MANY FILES THAT DO THIS IN A DIFFERENT WAY USING DIFFERENT METHODS
    """

import re
import nltk
import spacy
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


def read_tsv(filepath):
    """
    read in a tsv file
    """
    return pd.read_csv(filepath, sep="\t")


def make_q_and_a_col(df):
    """
    Makes a column that concatenates the strings
    from the question and answer columns
    Args:
        df (Pandas DataFrame):
    Returns:
        Pandas DataFrame with an additional column
    """
    df['question_and_answer'] = df["question"] + ' ' + df['answer']
    return df


def remove_punc2(texts):
    x = [''.join(c for c in s if c not in string.punctuation) for s in texts]
    x = [s for s in x if s]
    return x


def remove_punc(texts):
    '''
    texts are a list of a string where each string is a row of text from the df
    '''
    clues = [re.sub('\S*@\S*\s?', '', sent) for sent in texts]

    # Remove new line characters
    clues = [re.sub('\s+', ' ', sent) for sent in clues]

    # Remove distracting single quotes
    clues = [re.sub("\'", "", sent) for sent in clues]
    return clues


# TODO:
# Get this to function as a tokenizer to use in the Vectorizer of clustering_model_1
def tokenize(texts):
    for sent in texts:
        yield(gensim.utils.simple_preprocess(str(texts), deacc=True))


def make_bigrams(texts):
    words = list(tokenize(texts))
    bigram = gensim.models.Phrases(words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    words = list(tokenize(texts))
    trigram = gensim.models.Phrases(bigram[words], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def tokenize(text):
    """[summary]
    Args:
    text (string): a string to be tokenized
    Returns:
    a list of strings: tokenized words
    """
    lemmatizer = WordNetLemmatizer()
    # stemmer = SnowballStemmer('english')
    word_list = word_tokenize(text)

    lemmatized_wrds = [lemmatizer.lemmatize(w) for w in word_list]
    # stemmed_wrds = [stemmer.stem(w) for w in lemmatized_wrds]
    # return stemmed_wrds
    return lemmatized_wrds


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


def remove_stopwords(texts, stop_words):
    '''
    texts need to be tokenized
    '''
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# TODO: maybe I don't need this one anymore
def remove_stopwords_to_df(df, col, stop_words):
    """
    remove stopwords from a set
    Args:
        df (Pandas DataFrame)
        cols (list of str): list of columns to be cleaned, 
                            as strings
    Returns:
        Pandas Dataframe: the original dataframe with the
        column without stopwords
    """
    df[col] = df[col].apply(lambda x: ' '.join(
        [word for word in simple_preprocess(str(x)) if word not in stop_words]))
    return df


def preprocess_columns(text, stop_words):
    '''
    text = the tolist() from column value
    '''
    clues = remove_punc(text)
    clues = remove_stopwords(clues, stop_words)
    clues = make_bigrams(clues)
    clues = lemmatization(clues)
    clues = remove_stopwords(clues, stop_words)
    return clues


if __name__ == "__main__":

    regular_episodes = pd.read_csv('../data/regular_episodes.csv')
    # # Create stopwords list

    text = convert_col_to_list(regular_episodes)

