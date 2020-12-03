import numpy as np
import matplotlib.pyplot as plt

import preprocessing 
import warnings
warnings.filterwarnings('ignore')


from sklearn.decomposition import NMF
# for LDA
from sklearn.feature_extraction.text import CountVectorizer
# for NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# vectorize the corpus
stopwords = preprocessing.make_stopwords(None)
n_features = None


# calculate the feature matrix

# #TODO: turn into a function with df, col
def make_tfidf(df, col):
    tfidf_vectorizer = TfidfVectorizer(min_df=10, max_df=0.95, ngram_range=(1,1), stop_words=stopwords, max_features = n_features)
    tfidf_feature_matrix = tfidf_vectorizer.fit_transform(df[col])

    nmf = NMF(n_components=2, random_state=43,  alpha=0.1, l1_ratio=0.5)
    nmf.fit_transform(tfidf_feature_matrix)
    # nmf_output = nmf.fit_transform(tfidf_feature_matrix)

    nmf_feature_names = tfidf_vectorizer.get_feature_names()
    nmf_weights = nmf.components_
    return nmf_feature_names, nmf_weights


# get topics with their terms and weights
def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]

    return topics

# prints components of all the topics
# obtained from topic modeling
def print_topics_udf(topics, total_topics,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):

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

# prints components of all the topics
# obtained from topic modeling

def get_topics_udf(topics, total_topics,
                     weight_threshold=0.0001,
                     num_terms=None):

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

def getTermsAndSizes(topic_display_list_item):
    """

    Args:
        topic_display_list_item (list item): an element indexed 
            from the topic_display_list that is returned from 
            the get_topics_udf function 

    Returns:
        [tupe of lists]: [description]
    """    
    terms = []
    sizes = []
    for term, size in topic_display_list_item:
        terms.append(term)
        sizes.append(size)
    return terms, sizes


#TODO: turn into a function for each latent topic 
# add back topics_lst to the parameters if looping
def show_impt_words(n_top_words, n_topics, save = False):
    """[summary]

    Args:
        topics_lst ([type]): [description]
        n_top_words ([type]): [description]
        n_topics ([type]): [description]
        save (bool, optional): [description]. Defaults to False.
    """    
    # for i in topics_lst:
    #     terms, sizes = getTermsAndSizes(topics_display_list[i])

    terms, sizes = getTermsAndSizes(topics_display_list[1])
    fontsize_base = n_top_words / np.max(sizes) # font size for word with largest share in corpus

    for t in range(n_topics):
        fig, ax = plt.subplots(1, n_topics, figsize=(6, 12))
        plt.ylim(0, n_top_words + 1.0)
        plt.xticks([])
        plt.yticks([])
        plt.title('Topic #{}'.format(t))

        for i, (word, share) in enumerate(zip(terms, sizes)):
            word = word + " (" + str(share) + ")"
            plt.text(0.3, n_top_words-i-1.0, word, fontsize=fontsize_base*share)
    if save:
        plt.tight_layout()
        plt.savefig(f'../images/impt_words_{i}')
    else:
        plt.tight_layout()
        plt.show()

# def wordcloud_impt_wrds()

#     #generate word list
#     word_lst = terms
#     words = ' '.join(terms)
#     wordcloud = WordCloud(width = 800, height = 800, 
#                 background_color =None, mode = 'RGBA', 
#                 colormap = color,
#                 collocations=False,
#                 min_font_size = 10).generate(words) 

#     # plot the WordCloud image                        
#     plt.figure(figsize = (8, 8), facecolor = None) 
#     plt.imshow(wordcloud) 
#     plt.axis("off") 
#     plt.tight_layout(pad = 0) 

#     if save:
#         plt.savefig(f'../images/eda_images/{col}_wordcloud.png')
#     else:
#         plt.show()


if __name__ == "__main__":

    jeopardy_df = preprocessing.read_tsv('../data/master_season1-35.tsv')
    # jeopardy_df = clean_columns(jeopardy_df, ['category', 'comments', 'answer', 'question'])
    jeopardy_df = preprocessing.update_df_columns(jeopardy_df)
    regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
    special_tournaments = jeopardy_df.drop(regular_episodes.index)

    regular_episodes_sub = preprocessing.make_sub_df(regular_episodes)
    df = regular_episodes_sub
    col = 'question_and_answer'

    n = 2

    

    # tfidf_vectorizer = TfidfVectorizer(min_df=10, 
    #                     max_df=0.95, ngram_range=(1,1),
    #                     stop_words=stopwords, max_features = n_features)

    # tfidf_feature_matrix = tfidf_vectorizer.fit_transform(df[col])

    # nmf = NMF(n_components=2, random_state=43,  alpha=0.1, l1_ratio=0.5)
    # nmf.fit_transform(tfidf_feature_matrix)
    # mf_output = nmf.fit_transform(tfidf_feature_matrix)

    # nmf_feature_names = tfidf_vectorizer.get_feature_names()
    # nmf_weights = nmf.components_
   

    # n = 2
    # nmf_feature_names, nmf_weights = make_tfidf(df, col)
    # topics = get_topics_terms_weights(nmf_weights, nmf_feature_names)  
    # print_topics_udf(topics, total_topics= n, num_terms=30, display_weights=True)
    # topics_display_list = get_topics_udf(topics, total_topics=n, num_terms=30)

    # topics_udf = get_topics_udf(topics, total_topics=1,
    #                  weight_threshold=0.0001,
    #                  num_terms=None)

    # terms, sizes = getTermsAndSizes(topics_display_list[0])

    # n_top_words = 20
    # n_topics = 1
    # # show_impt_words(n_top_words, 1, save = False)
