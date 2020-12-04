import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import preprocessing


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
    """[summary]

    Args:
        df ([type]): [description]
        col ([type]): [description]
        vectorizer ([type]): [description]
        n_topics ([type]): [description]
        nmf ([type]): [description]

    Returns:
        [type]: [description]
    """    
    tfidf = vectorizer.fit_transform(df[col])

    nmf.fit_transform(tfidf) # W matrix 
    nmf_feature_names = vectorizer.get_feature_names()  #Feature names 
    nmf_weights = nmf.components_ #H
    recon_err = nmf.reconstruction_err_
    return nmf_feature_names, nmf_weights, recon_err

def plot_ks(df, col, vectorizer, n_topics, nmf):
    errs = []
    for k in range(n_topics):
        nmf_feature_names, nmf_weights, recon_err = get_names_weights(df, col, vectorizer, n_topics, nmf)
        errs.append(recon_err)
    plt.plot(range(1,n_topics), errs)
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.show();
    return errs

def make_wrds_topics(feature_names, weights, n_topics, n_top_words, vectorizer, nmf):
    """
    Takes an input of feature names, or words, and their weights
    and returns the top n words, the latent topic index
    and the reconstruction error of the nmf model 
    
    Args:
        feature_names ([type]): [description]
        weights ([type]): [description]
        n_topics ([type]): [description]
        n_top_words ([type]): [description]
        vectorizer ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    words = []
    topic_indices = []
    for topic_idx, topic in enumerate(weights):
        words.append(list(feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]))
        # names.append(" ".join([feature_names[i]
        #             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        topic_indices.append(topic_idx)
    
    return words, topic_indices



# get topics with their terms and weights
def get_topics_terms_weights(feature_names, weights):
    """[summary]

    Args:
        feature_names ([type]): [description]
        weights ([type]): [description]

    Returns:
        [type]: [description]
    """    
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]
    return topics, sorted_weights

def make_weights_dict(topics, nth_topic, n_top_words):
    """[summary]

    Args:
        n_topics ([type]): [description]
        n_top_words ([type]): [description]
        sorted_weights ([type]): [description]

    Returns:
        [type]: [description]
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

    
def viz_top_words(dictionary, color, n, save = False): 
    wordcloud = WordCloud(width=800,height=800, 
                        relative_scaling=.5, normalize_plurals=True,
                        background_color =None, mode = 'RGBA', 
                        colormap = color, collocations=False, 
                        min_font_size = 10)
    
    wordcloud.generate_from_frequencies(dictionary)

    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    if save:
        plt.savefig(f'../images/{n}topic_model_Wordcloud.png')
    else:
        plt.show()

def show_word_clouds(n_topics, topics, n_top_words, color = 'plasma', save= False):
    """[summary]

    Args:
        n_topics ([type]): [description]
        topics ([type]): [description]
        n_top_words ([type]): [description]
        color (str, optional): [description]. Defaults to 'plasma'.
        save (bool, optional): [description]. Defaults to False.
    """    
    for nth_topic in range(n_topics):
        dictionary = make_weights_dict(topics, nth_topic, n_top_words)
        viz_top_words(dictionary, n = nth_topic, color = 'plasma', save = save)


if __name__ == "__main__":

    regular_episodes = pd.read_csv("../data/jeopardy_regular_episodes.csv")
    regular_episodes_sub = preprocessing.make_sub_df(regular_episodes)

    regular_episode_sub_reindexed = regular_episodes_sub.set_index('J-Category')
    regular_episodes_reindexed = regular_episodes.set_index('J-Category')

    # Use the model 
    stopwords = preprocessing.make_stopwords(None) #
    df = regular_episodes_reindexed
    col = 'Question and Answer'

    #adjust these hyperparameters
    n_topics = 13
    n_top_words = 10
    tot_features = 1000

    #Adjust the vectorizer and nmf model hyperparameters 

    nmf = NMF(n_components=n_topics, random_state=43,  
                    alpha=0.1, l1_ratio=0.5)

    vectorizer = TfidfVectorizer(
                    ngram_range=(1,2), strip_accents = 'ascii',
                    lowercase = True, tokenizer = preprocessing.tokenize,
                    analyzer = 'word', stop_words= stopwords,
                    max_features = tot_features)

    feature_names, weights, recon_err = get_names_weights(df, col, vectorizer, n_topics, nmf)
    topics, sorted_weights = get_topics_terms_weights(feature_names, weights)

    words, topic_indices = make_wrds_topics(feature_names, weights, n_topics, n_top_words, vectorizer, nmf)

    # for i in range(n_topics):
    #     print (f'Topic {i+1}')
    #     for l in topics[i][:10]:
    #         print (l)
    #     print ('\n')
    # print (f'Reconstruction Error: {recon_err}')

    show_word_clouds(n_topics, topics, n_top_words, color = 'plasma', save= True)