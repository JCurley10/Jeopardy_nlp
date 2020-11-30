import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Episode(object):

    def __init__(self, df, regular_episode, easy_tournament, hard_tournament):
        self.all_episodes = df
        self.regular_episode = self.filter_tournament('regular')
        self.easy_tournament = self.filter_tournament('easy')
        self.hard_tournament = self.filter_tournament('hard')

    def filter_tournament(self, difficulty):
        """[summary]

        Args:
            difficulty (string): options: 'regular', 'easy', 'hard',
                depending on the kind of episode I want to look at

        Returns:
            Pandas DataFrame: a subset of the original dataframe 
                that only includes episides from the given difficulty 
        """        

        if difficulty =='regular':
            self.regular_episode = self.all_episodes[self.all_episodes['notes']=='-']
            return self.regular_episode
        elif difficulty == 'easy':
            self.easy_tournament = self.all_episodes[self.all_episodes['notes'].str.contains('kids', 'teen', 'celebrity', 'college')]
            return self.easy_tournament
        elif difficulty == 'hard':
            self.hard_tournament = self.all_episodes[self.all_episodes['notes'].str.contains('championship')]
            return self.hard_tournament

if __name__ == "__main__":

    jeopardy = pd.read_csv('../data/master_season1-35.tsv', sep = "\t")

    jeopardy_episode = Episode(jeopardy)
    jeopardy_episode.all_episodes # passes back all data
    jeopardy_episode.regular_episode # should be the same as above 