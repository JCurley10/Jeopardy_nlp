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
    """[summary]

    Args:
        df ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    count_vect = CountVectorizer(ngram_range = (1, 1), 
                            lowercase=True,  tokenizer=None, 
                            stop_words= stopwords, analyzer='word',  
                            max_features=None)

    x = count_vect.fit_transform(df[col])
    # features = count_vect.get_feature_names()
    kmeans = KMeans(n_clusters = 10, random_state = 123).fit(x)
    centroids = kmeans.cluster_centers_
    top_n = np.argsort(centroids)[:, :-n+1:-1]
    names = count_vect.get_feature_names()

    name_arr = np.array(names)
    return f'n = {n}', name_arr[top_n]




#TODO: change the hyperparameters in the tfidf vectorizer or have the option
#TODO: write the docstring

def get_names_weights(df, col, vectorizer, n_topics, nmf):
    """
    words and weight from the get_names_weights function
    return an array of the words, or feature_names

    Args:
        df ([type]): [description]
        col ([type]): [description]
        vectorizer ([type]): [description]
        components (int): number of topics to seaprate it into

    Returns:
        [type]: [description]
    """    
    tfidf = vectorizer.fit_transform(df[col])

    W = nmf.fit_transform(tfidf) # W matrix 
    nmf_feature_names = vectorizer.get_feature_names()  #Feature names 
    nmf_weights = nmf.components_ #H
    return nmf_feature_names, nmf_weights


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
    reconstruction_error =  nmf.reconstruction_err_
    words = []
    topic_indices = []
    for topic_idx, topic in enumerate(weights):
        words.append(list(feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]))
        # names.append(" ".join([feature_names[i]
        #             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        topic_indices.append(topic_idx)
    
    return words, topic_indices, reconstruction_error



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

def make_weights_lst(topics, nth_topic, n_top_words):
    """[summary]

    Args:
        n_topics ([type]): [description]
        n_top_words ([type]): [description]
        sorted_weights ([type]): [description]

    Returns:
        [type]: [description]
    """   
    top_ten_topics = []
    for topic in topics:
        top_ten_topics.append(topic[:10])

    top_10_topics = []
    # for i in range(n_topics):
    #     for j in range(n_top_words):
    #         top_10_topics.append(top_ten_topics[i][j][0])
    #         top_10_topics.append(float(top_ten_topics[i][j][1]))

    # weights_dictionary = dict(zip(top_10_topics[::2], top_10_topics[1::2]))
    # return weights_dictionary

    i = nth_topic
    for j in range(n_top_words):
        top_10_topics.append(top_ten_topics[i][j][0])
        top_10_topics.append(float(top_ten_topics[i][j][1]))

    weights_dictionary = dict(zip(top_10_topics[::2], top_10_topics[1::2]))
    return weights_dictionary

    
def viz_top_words( dictionary, color, save = False): 
    wordcloud = WordCloud(width=800,height=800, 
                        max_words=20,relative_scaling=1,normalize_plurals=True,
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
        plt.savefig(f'../images/eda_images/{col}_wordcloud.png')
    else:
        plt.show()



if __name__ == "__main__":

    regular_episodes = pd.read_csv("../data/jeopardy_regular_episodes.csv")
    regular_episodes_sub = preprocessing.make_sub_df(regular_episodes)

    regular_episode_sub_reindexed = regular_episodes_sub.set_index('J-Category')
    regular_episodes_reindexed = regular_episodes.set_index('J-Category')

    # Use the model 
    stopwords = preprocessing.make_stopwords(None) #need to adjust
    df = regular_episode_sub_reindexed
    col = 'Question and Answer'

    #adjust these hyperparameters
    n_topics = 10
    n_top_words = 10
    tot_features = 1000

    #Adjust the vectorizer and nmf model hyperparameters 
    vectorizer = TfidfVectorizer(min_df=1, max_df=1.0,
                    ngram_range=(1,2), 
                    lowercase = True, 
                    analyzer = 'word', stop_words=stopwords,
                    max_features = tot_features)

    nmf = NMF(n_components=n_topics, random_state=43,  
                    alpha=0.1, l1_ratio=0.5)


    feature_names, weights =  get_names_weights(df, col, vectorizer, n_topics, nmf)
    topics, sorted_weights = get_topics_terms_weights(feature_names, weights)

    words, topic_indices, recon_err = make_wrds_topics(feature_names, weights, n_topics, n_top_words, vectorizer, nmf)

    nth_topic = 5
    dictionary = make_weights_lst(topics, nth_topic, n_top_words)
    # viz_top_words(dictionary, 'plasma', save = False)
    

    for i in range(n_topics):
        print (f'Topic {i+1}')
        for l in topics[i][:10]:
            print (l)
        print ('\n')
    print(f"reconstruction error: {recon_err}")
