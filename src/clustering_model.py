import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from preprocessing import tokenize, make_sub_df, make_stopwords


#TODO: hyperparameter tuning with the Count Vectorizer 
#TODO: write docstring

def kmeans_cluster(df, col, n, stopwords):
    """
    hard cluster of the topics in a given
    column of a dataframe using k-means

    Args:
        df (Pandas DataFrame): DataFrame with the top categories 
        col (str): column name to get the clusters 
        n (int): latent topic count

    Returns: 
        tuple that classifies the number of clusters, n
        and the top n words in the cluster
    
    """
    count_vect = CountVectorizer(ngram_range = (1, 1), 
                            lowercase=True,  tokenizer=None, 
                            stop_words= stopwords, analyzer='word',  
                            max_features=None)

    x = count_vect.fit_transform(df[col])
    # features = count_vect.get_feature_names()
    kmeans = KMeans(n_clusters = n, random_state = 123).fit(x)
    centroids = kmeans.cluster_centers_
    top_n = np.argsort(centroids)[:, :-n+1:-1]
    names = count_vect.get_feature_names()

    name_arr = np.array(names)
    return f'n = {n}', name_arr[top_n]


def get_names_weights(df, col, vectorizer, n_topics, nmf):
    """
    get the feature names, weights, and reconstruction
    error of the nmf model

    Args:
        df (Pandas DataFrame): DataFrame with the top categories
        col (str): column name to get the clusters
        vectorizer (type TfidfVectorizer() vectorizer): method to vectorize
                    the text
        n_topics (int): total number of topics to get clusters
        nmf (type sklearn NMF decomposer): initialized NMF instance

    Returns:
        tuple: feature names, weights, and reconstruction
        error of the nmf model
    """
    tfidf = vectorizer.fit_transform(df[col])

    nmf.fit_transform(tfidf) # W matrix 
    nmf_feature_names = vectorizer.get_feature_names()  # Feature names 
    nmf_weights = nmf.components_ # H
    recon_err = nmf.reconstruction_err_
    return nmf_feature_names, nmf_weights, recon_err


#TODO: get this function to work on large df without it timing out. 
#TODO: use a different function to get recon_err since it has to unpack the get_names_weights and doens't use two variables. 
def plot_ks(df, col, vectorizer, n_topics, nmf):
    """
    plot the number of topics vs the reconstruction error
    to look at the elbow and decide on which number
    of topics is best

    Args:
        df (Pandas DataFrame): DataFrame with the top categories 
        col (str): column name to get the clusters 
        vectorizer (type TfidfVectorizer() vectorizer): method to vectorize the text
        n_topics (int): total number of topics to get clusters of 
        nmf (type sklearn NMF decomposer): initialized NMF instance
    Returns:
        None: 
    """    
    errs = []
    for _ in range(n_topics):
        nmf_feature_names, nmf_weights, recon_err = get_names_weights(df, col, vectorizer, n_topics, nmf)
        errs.append(recon_err)
    plt.plot(range(1, n_topics), errs)
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.show()


def make_wrds_topics(feature_names, weights, n_topics, n_top_words, vectorizer, nmf):
    """
    Takes an input of feature names, or words, and their weights
    and returns the top n words, the latent topic index
    and the reconstruction error of the nmf model
    
    Args:
        feature_names (Series of str): the feature_names (words),
             the result[0] from the get_names_weights function
        weights (Series of floats): the weights of each feature (word)
            from feature_names. Result[1] from the get_names_weights function
        n_topics (int): total number of topics to get clusters of
        n_top_words (int): number of words per cluster
        vectorizer (type TfidfVectorizer() vectorizer): method to vectorize
                    the text

    Returns:
        a tuple of lists of a list of list of strings of n_top_words (feature_names) 
        and the topic indices in a list equal to the length of n_topics
    """    
    
    words = []
    topic_indices = []
    for topic_idx, topic in enumerate(weights):
        words.append(list(feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]))
        topic_indices.append(topic_idx)
    
    return words, topic_indices


def get_topics_terms_weights(feature_names, weights):
    """
    Get the feature names (words) for each cluster, and the sorted weights

    Args:
        feature_names (Series of str): the feature_names (words),
             the result[0] from the get_names_weights function
        weights (Series of floats): the weights of each feature (word)
            from feature_names. THe result[1] from the get_names_weights function

    Returns:()
        tuple
        topics: list of lists  of strings
             the feature_names (or words) with their weights as strings
        sorted_weights: Numpy Array of weights, in descending order,
            associated with each vector associated with its corresponding
            list index in feature_names
    """
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]
    return topics, sorted_weights


def make_weights_dict(topics, nth_topic, n_top_words):
    """
    make a dictionary where the keys are the words or feature_names
    and the value are the associate weights. This will be used
    for graphing a wordcloud generated by word frequencies
    and the weights will be the frequencies 

    Args:
        topics (list of lists  of strings): the feature_names 
            (or words) with their weights as strings
        n_topics (int): total number of topics to get clusters of 
        n_top_words (int): number of words per cluster
        sorted_weights: Numpy Array of weights, in descending order,
            associated with each vector associated with its corresponding 
            list index in feature_names 

    Returns:
        Dictionary: the feature_names or words are the keys and the 
        values are the associated weights 
    """
    top_n_topics = []
    for topic in topics:
        top_n_topics.append(topic[:10])

    top_10_topics = []
    i = nth_topic
    for j in range(n_top_words):
        top_10_topics.append(top_n_topics[i][j][0])
        top_10_topics.append(np.sqrt(float(top_n_topics[i][j][1])))

    weights_dictionary = dict(zip(top_10_topics[::2], top_10_topics[1::2]))
    return weights_dictionary


def viz_top_words(dictionary, color, n, save=False): 
    """
    Make a wordcloud of the feature_names (words) 
    from a single cluster or topic

    Args:
        dictionary ([type]): [description]
        color (str): matplotlib colormap
        n (int): The topic number to make a wordcloud of
        save (bool, optional): If True, saves figure to filepath
            If False, shows figure. Defaults to False.
    Returns:
        None. Generates a wordcloud, and either saves or shows it
    """    
    wordcloud = WordCloud(width=800,height=800, 
                        relative_scaling=.5, normalize_plurals=True,
                        background_color=None, mode='RGBA', 
                        colormap=color, collocations=False, 
                        min_font_size=10)
    
    wordcloud.generate_from_frequencies(dictionary)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None) 
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    if save:
        plt.savefig(f'../images/{n}topic_model_Wordcloud.png')
    else:
        plt.show()

def show_word_clouds(n_topics, topics, n_top_words, color = 'plasma', save= False):
    """
    Show or save the wordclouds for all topics 

    Args:
        n_topics (int): total number of topics to get clusters of 
        topics: list of lists  of strings
             the feature_names (or words) with their weights as strings
        n_top_words (int): number of words per cluster
        color (str, optional): matplotlib color. Defaults to 'plasma'.
        save (bool, optional): If True, saves figure to filepath
            If False, shows figure. Defaults to False.
    Returns:
        None. saves or shows the wordclouds
    """    
    for nth_topic in range(n_topics):
        dictionary = make_weights_dict(topics, nth_topic, n_top_words)
        viz_top_words(dictionary, n = nth_topic, color = 'plasma', save = save)


if __name__ == "__main__":

    regular_episodes = pd.read_csv("../data/jeopardy_regular_episodes.csv")
    regular_episodes_sub = make_sub_df(regular_episodes)

    # To look at W and H with respect to the original J-categories
    regular_episode_sub_reindexed = regular_episodes_sub.set_index('J-Category')
    regular_episodes_reindexed = regular_episodes.set_index('J-Category') # get a smaller samplle 

    stopwords = make_stopwords(None) #get the stopwords from the preprocessing file
    df = regular_episodes_reindexed
    col = 'Question and Answer'

    # Adjust these hyperparameters
    n_topics = 13
    n_top_words = 10
    tot_features = 1000

    # Adjust the vectorizer and nmf model hyperparameters 
    nmf = NMF(n_components=n_topics, random_state=43,  
                    alpha=0.1, l1_ratio=0.5)

    vectorizer = TfidfVectorizer(
                    ngram_range=(1,2), strip_accents = 'ascii',
                    lowercase = True, tokenizer = tokenize,
                    analyzer = 'word', stop_words= stopwords,
                    max_features = tot_features)

    feature_names, weights, recon_err = get_names_weights(df, col, vectorizer, n_topics, nmf)
    topics, sorted_weights = get_topics_terms_weights(feature_names, weights)
    words, topic_indices = make_wrds_topics(feature_names, weights, n_topics, n_top_words, vectorizer, nmf)

    print(plot_ks(regular_episodes, col, vectorizer, 25, nmf))