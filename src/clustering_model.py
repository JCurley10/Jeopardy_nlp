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

def kmeans_cluster(df, col, n):
    """[summary]

    Args:
        df ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    stopwords = make_stopwords(None)    
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
def nm_factorize(df, col, n_features, n_topics, n_top_words):
    """[summary]

    Args:
        df ([type]): [description]
        col ([type]): [description]
        n_features ([type]): [description]
        n_topics ([type]): [description]

    Returns:
        [type]: [description]
    """ 
    stopwords = make_stopwords(None) 
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words=stopwords) #adjust stopwords
                             
    tfidf = vectorizer.fit_transform(df[col])

    nmf = NMF(n_components=n_topics, random_state=123)
    nmf.fit(tfidf)

    W = nmf.transform(tfidf)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names()
    weights = nmf.components_

    topics = ['latent_topic_{}'.format(i) for i in range(n_topics)]
    idx = df.index
    col = vectorizer.vocabulary_.keys()

    W = pd.DataFrame(W, index = idx, columns = topics)
    H = pd.DataFrame(H, index = topics, columns = col)

    W,H = (np.around(x, 2) for x in (W, H))

    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    print (f'RECONSTRUCTION: {nmf.reconstruction_err_}')
    # print ()
    # print (W.head(30), '\n\n', H.head(n_topics))

    return None

"""----------------"""


def nm_factorizer(df, col, n_features, total_topics, ):
    """[summary]

    Args:
        df ([type]): [description]
        col ([type]): [description]
        n_features (int): number of components or feature
        n_topics ([type]): [description]

    Returns:
        [type]: [description]
    """ 
    stopwords = make_stopwords(None) 
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words=stopwords) #adjust stopwords
                             
    tfidf = vectorizer.fit_transform(df[col])

    nmf = NMF(n_components=total_topics, random_state=123)
    nmf.fit(tfidf)

    feature_names = vectorizer.get_feature_names()
    weights = nmf.components_
    return weights, feature_names


def get_term_weights(weights, feature_names):
    """
    returns the topics with the terms and weights

    Args:
        weights ([type]): [description]
        feature_names ([type]): [description]

    Returns:
        [type]: [description]
    """    
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]

    return topics

def print_topic_terms(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):
    """
    prints the components of topics 
    returned from get_term_weights

    Args:
        topics ([type]): [description]
        total_topics (int, optional): [description]. Defaults to 1.
        weight_threshold (float, optional): [description]. Defaults to 0.0001.
        display_weights (bool, optional): [description]. Defaults to False.
        num_terms ([type], optional): [description]. Defaults to None.
    """                     

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        #print(topic)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms]) if num_terms else topic
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms]) if num_terms else tw


def return_topic_terms(topics, total_topics=1,
                     weight_threshold=0.0001,
                     num_terms=None):
    """
    returns a list of terms from the topics 
    obtained from get_term_weights

    Args:
        topics ([type]): [description]
        total_topics (int, optional): [description]. Defaults to 1.
        weight_threshold (float, optional): [description]. Defaults to 0.0001.
        num_terms ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """                     

    topic_terms = []

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        #print(topic)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        topic_terms.append(topic[:num_terms] if num_terms else topic)

    return topic_terms

def get_terms_sizes(topic_display_list_item):
    """
    Returns the list of terms and sizes 

    Args:
        topic_display_list_item (list): topic_terms
            output from the return_topic_terms function 

    Returns:
        [tuple of lists]: terms and sizes lists, 
            that come from the topic_terms list

    """    
    terms = []
    sizes = []
    for term, size in topic_display_list_item:
        terms.append(term)
        sizes.append(size)
    return terms, sizes

if __name__ == "__main__":

    jeopardy_df = read_tsv('../data/master_season1-35.tsv')
    # jeopardy_df = clean_text(jeopardy_df, ['category', 'comments', 'answer', 'question'])
    jeopardy_df = update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)

    regular_episode_sub = make_sub_df(regular_episodes)
    regular_episode_sub_reindexed = regular_episode_sub.set_index('category')
    regular_episodes_reindexed = regular_episodes.set_index('category')

  



    total_topics = None

    weights, feature_names = nm_factorizer(regular_episode_sub_reindexed, 'question_and_answer', n_features, n_topics, n_top_words)
    topics = get_term_weights(weights, feature_names)
    print_topic_terms(topics, total_topics= 5, num_terms = 20, display_weights= True)


    
    
    # full_set = nm_factorizer(regular_episodes_reindexed, 'question_and_answer', n_features, n_topics, n_top_words)


