import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cleaning import read_tsv


def make_q_and_a_col(df):
    """
    Makes a column that concatenates the strings
    from the question and answer columns

    Args:
        df (Pandas DataFrame): 
    Returns:
        Pandas DataFrame with an additional column
    """    
    df['question_and_answer'] = df["question"] + df['answer']
    return df

#TODO: make another condition using "view assummptions" 
def make_q_difficulty_col(df):
    conditions = [((df['value']<=600) & (df['daily_double']=='no')), #easy
                ((df['daily_double']=='no') & ((df['value']==800) | (df['value']==1200))), #average
                ((df['daily_double']== 'yes') & (df['round'] == 1)), #average
                ((df['daily_double']=='no') & ((df['value']==1000) | (df['value']>=1600))), #hard
                ((df['daily_double']== 'yes') & (df['round'] == 2)), #hard
                (df['round'] == 3)] # final jeopardy, hard 


    difficulties = ['easy', 'average', 'average', 'hard', 'hard', 'hard']

    df['question_difficulty'] = np.select(conditions, difficulties)
    return df

#TODO: write docstring
def update_df_columns(df):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    df_new = make_q_and_a_col(df)
    df_new = make_q_difficulty_col(df_new)
    return df_new

if __name__ == "__main__":

