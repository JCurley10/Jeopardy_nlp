import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from preprocessing import read_tsv, update_df_columns, clean_text, clean_columns, make_sub_df



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
    count_vect = CountVectorizer(ngram_range = (1, 1), 
                            lowercase=True,  tokenizer=None, 
                            stop_words='english', analyzer='word',  
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

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words='english') #adjust stopwords
    tfidf = vectorizer.fit_transform(df[col])

    nmf = NMF(n_components=n_topics, random_state=123)
    nmf.fit(tfidf)

    W = nmf.transform(tfidf)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names()

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


if __name__ == "__main__":

    jeopardy_df = read_tsv('../data/master_season1-35.tsv')
    jeopardy_df = clean_columns(jeopardy_df, ['category', 'comments', 'answer', 'question'])
    jeopardy_df = update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)

    regular_episode_sub = make_sub_df(regular_episodes)
    regular_episode_sub_reindexed = regular_episode_sub.set_index('category')
    regular_episodes_reindexed = regular_episodes.set_index('category')

    # n_samples = 2000
    n_features = 100
    n_topics = 10
    n_top_words = 20

    nm_factorize(regular_episodes_reindexed, 'question_and_answer', n_features, n_topics, n_top_words)



