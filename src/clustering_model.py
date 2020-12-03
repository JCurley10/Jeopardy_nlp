import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from preprocessing import read_tsv, update_df_columns, clean_text, make_sub_df, make_stopwords



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
    """[summary]

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
    """[summary]

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
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]

    return topics


def viz_top_words(words, weights, color, save = False):
    # word_weight = None
    # d = dict()
    # scalar = 

    # # make a dictionary with d = {word: loadings*n}...
    # for word in words:
    #     d[word] = word_weight * scalar
    wordcloud = WordCloud(width=800,height=800, 
                        max_words=20,relative_scaling=1,normalize_plurals=True,
                        background_color =None, mode = 'RGBA', 
                        colormap = color, collocations=False, 
                        min_font_size = 10).generate_from_frequencies(d)

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

    jeopardy_df = preprocessing.read_tsv('../data/master_season1-35.tsv')
    jeopardy_df = preprocessing.lowercase(jeopardy_df, ['category'])
    jeopardy_df = preprocessing.remove_punc(jeopardy_df, ['category', 'question', 'answer'])
    jeopardy_df = preprocessing.update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)
    
    regular_episodes_sub = preprocessing.make_sub_df(regular_episodes)


    regular_episode_sub_reindexed = regular_episode_sub.set_index('category')
    regular_episodes_reindexed = regular_episodes.set_index('category')

    # Use the model 
    stopwords = make_stopwords(None) #need to adjust
    df = regular_episode_sub_reindexed
    col = 'question_and_answer'

    n_topics = 10
    n_top_words = 10
    tot_features = 1000

    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95,
                     ngram_range=(1,2), lowercase = True, 
                     analyzer = 'word', stop_words=stopwords,
                    max_features = tot_features)

    nmf = NMF(n_components=n_topics, random_state=43,  
                    alpha=0.1, l1_ratio=0.5)


    feature_names, weights =  get_names_weights(df, col, vectorizer, n_topics, nmf)
    topics = get_topics_terms_weights(feature_names, weights)

    words, topic_indices, recon_err = make_wrds_topics(feature_names, weights, n_topics, n_top_words, vectorizer, nmf)
    
    
    # vocab_lst = []
    # loadings_lst = []
    # for i in range(len(topics[:][:10])):
    #     vocab = topics[i][0]
    #     loadings = topics[i][0]

    for i in range(n_topics):
        print (f'Topic {i+1}')
        for l in topics[i][:10]:
            print (l)
        print ('\n')