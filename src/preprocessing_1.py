import re
import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.corpus import stopwords

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


def remove_punc(texts):
    cleaned_text = [re.sub('\S*@\S*\s?', '', sent) for sent in texts]
    cleaned_text = [re.sub('\s+', ' ', sent) for sent in cleaned_text]
    cleaned_text = [re.sub("\'", "", sent) for sent in cleaned_text]
    return cleaned_text


# TODO:
# Get this to function as a tokenizer to use in the Vectorizer of clustering_model_1
def tokenize(texts):
    for sent in texts:
        yield(gensim.utils.simple_preprocess(str(texts), deacc=True))


def make_bigrams(texts):
    bigram = gensim.models.Phrases(clue_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    trigram = gensim.models.Phrases(bigram[clue_words], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def make_stopwords(filepath):
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


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def remove_stopwords_to_df(df, col, stopwords):
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
        [word for word in simple_preprocess(str(x)) if word not in stopwords]))
    return df


def remove_punc(df, col):
    """
    remove punctuation from a column
    Args:
        df (Pandas dataFrame): The dataframe in use
        col (str): the column name to turn into a string
    Returns:
        Pandas DataFrame: with removed punctuation
    """
    for col in cols:
        p = re.compile(r'[^\w\s]+')
        df[col] = [p.sub('', x) for x in df[col].tolist()]
    return df


def make_bigrams(texts):
    bigram = gensim.models.Phrases(clue_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


if __name__ == "__main__":

    regular_episodes = pd.read_csv('../data/regular_episodes.csv')
    # # Create stopwords list
    stopwords_txt = "stopwords.txt"
    stopwords_list = make_stopwords(stopwords_txt)
