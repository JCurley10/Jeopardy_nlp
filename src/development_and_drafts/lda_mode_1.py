from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing_1 import make_stopwords


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


if __name__ == "__main__":
    regular_episodes = pd.read_csv('../data/regular_episodes.csv')

    text = regular_episodes['Question and Answer'].values.tolist()
    stopwords = make_stopwords()
    n_samples = 2000
    n_features = 1000
    n_components = 10
    n_top_words = 10

    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=5,
                                    max_features=n_features,
                                    stop_words=stopwords, strip_accents='ascii', 
                                    ngram_range=(1,2))

    tf = tf_vectorizer.fit_transform(text)

    lda = LatentDirichletAllocation(n_components=n_components)

    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')