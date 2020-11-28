import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Difficulty:
    HARD = "HARD"
    AVERAGE = "AVERAGE"
    EASY = "EASY"

class Clue:
    
    def __init__(self, game_round, clue_value, daily_double, category, comments, answer, question, air_date, notes):
        self.game_round = game_round
        self.clue_value = clue_value
        self.daily_double = daily_double
        self.category = category
        self.comments = comments
        self.answer = answer
        self.question = question
        self.air_date = air_date
        self.notes = notes
        self.clue_difficulty = self.get_clue_difficulty
        self.game_difficulty = self.get_game_difficulty

    def get_clue_difficulty(self):
        """
        checks if a question is under or over a threshold
        and reutrns a string of the difficulty level
        """
        if (self.game_round == 1 and self.clue_value <= 600) or (self.game_round == 2 and self.clue_value <= 1200):
                return Difficulty.EASY
        else:
            return Difficulty.HARD


    def get_game_difficulty(self):
        """
        Checks if the game is from a regular episode or a special tournament
        If the game is regular, the difficulty is avergae
        if the game is from kids/teens/college, the diffuclty is easy
        if the game is from a championship, the difficulty is hard
        """        
        if self.notes == '-':
            return Difficulty.AVERAGE
        #TODO: if notes contains words for kids/teen/college/celebrity/ it's easy
        elif self.notes == None:
            return None
        ##TODO: if notes contains champions/
        elif self.notes == None:
            return None



if __name__ == "__main__":
    #train all
    #test all
    #sub train all 
    #train regular 
    #test regular
    #sub train regular  
    #train special
    #test special 
    #sub train regular 

    #TODO: fix up these training and testing sets 
    jeopardy = pd.read_csv('../data/master_season1-35.tsv', sep = "\t")
    regular_tournament = jeopardy[jeopardy['notes']=='-']
    special_tournament = jeopardy.drop(regular_tournament.index)

    train_set_all  = jeopardy.sample(frac = .8, axis = 0, random_state = 123)
    test_set_all = jeopardy.drop(train_set_all.index) 
    sub_train_all = train_set_all.sample(frac = .01, axis = 0, random_state = 123)
    
    train_reg = regular_tournament.sample(frac = .8, axis = 0, random_state = 123)  #just the questions from a regular episode
    test_reg = regular_tournament.drop(train_reg.index)
    train_reg_sub = train_reg.sample(frac = 0.01, axis = 0, random_state = 123) #a subsample from the training set of regular episodes
    
    train_spec = special_tournament.sample(frac = .8, axis = 0, random_state = 123) #questions from a special episodes/tournaments 
    test_spec = special_tournament.drop(train_spec.index)
    train_spec_sub = train_spec.sample(frac = 0.01, axis = 0, random_state = 123) #a subsample from the training set of special espisodes/tournaments

    #TODO: clean this up so I can get all the info per row to do a bag of words
    df = train_reg_sub
    clues = []
    for index, rows in df.iterrows():
        clues.append(Clue(df['round'], 
        df['value'], df['daily_double'], 
        df['category'], df['comments'],
        df['answer'], df['question'], 
        df['air_date'], df['notes']))

    #TODO: make a new row that combines answer and question 

    # what categories go together ?
    #replace the indices with the jep.category and see which jep.categories are clustered together 