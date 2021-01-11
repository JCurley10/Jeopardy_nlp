import re
import nltk
import spacy
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_tsv(filepath):
    """
    read in a tsv file to a pandas dataframe
    """
    return pd.read_csv(filepath, sep="\t")


def rename_columns(df):
    """
    Fix up the capitalization and underscores in the
    column names of the dataframe
    """
    new_df = df.rename(columns={"round": "Round",
                                "value": "Value",
                                "category": "J-Category",
                                "answer": "Answer",
                                "question": "Question",
                                "daily_double": "Daily Double",
                                "air_date": "Air Date"})
    return new_df


def make_q_and_a_col(df):
    """
    Makes a column that concatenates the strings
    from the question and answer columns
    -------
    Args:
        df (Pandas DataFrame):
    Returns:
        Pandas DataFrame with an additional column
    """
    df['Question And Answer'] = df["Question"] + ' ' + df['Answer']
    return df


def make_clue_difficulty_col(df, viewer_assumptions=False):
    """
    make a column of clue difficulty according to
    either viewer assumption or another analysis
    Args:
        df (Pandas DataFrame):
            viewer_assumptions (bool, optional): The jeopardy viewer assumption
            of what clues are hard. Defaults to False.
    Returns:
        Pandas DataFrame with question_diff column
    """
    if viewer_assumptions:
        conditions = [((df['Daily Double']=='no') & (df['Value'] <= 800)), #easy
            ((df['Daily Double']=='no') & (df['Round']== 1) & (df['Value'] >= 800)), #average
            ((df['Daily Double']=='no') & (df['Value'] == 1200)) #average
            ((df['Daily Double']=='no') & (df['Round']== 2) & (df['Value'] >= 1600)), #hard
            ((df['Daily Double']== 'yes') & (df['Round'] == 1)), #average
            ((df['Daily Double']== 'yes') & (df['Round'] == 2)), #hard
            (df['Round'] == 3)] # final jeopardy, hard 
        difficulties = ['easy', 'average', 'average', 'hard', 'average', 'hard', 'hard']

    else:
        conditions = [((df['Value'] <= 600) & (df['Daily Double'] == 'no')), # easy
            ((df['Daily Double'] == 'no') & ((df['Value'] == 800) | (df['Value'] == 1200))),  # average
            ((df['Daily Double'] == 'yes') & (df['Round'] == 1)),  # average
            ((df['Daily Double'] == 'no') & ((df['Value'] == 1000) | (df['Value'] >= 1600))),  # hard
            ((df['Daily Double'] == 'yes') & (df['Round'] == 2)),  # hard
            (df['Round'] == 3)]  # final jeopardy, hard

        difficulties = ['easy', 'average', 'average', 'hard', 'hard', 'hard']

    df['Clue Difficulty'] = np.select(conditions, difficulties)
    return df


def make_regular_episides_df(df):
    """
    Make a dataframe with just the regular Jeopardy! episodes
    """
    regular_episodes_df = df[df['notes'] == '-']
    regular_episodes_df = regular_episodes_df.drop(['notes', 'comments'], axis=1)
    regular_episodes_df = rename_columns(regular_episodes_df)
    regular_episodes_df = make_q_and_a_col(regular_episodes_df)
    # regular_episodes_df = make_clue_difficulty_col(regular_episodes_df)
    return regular_episodes_df


def make_special_tournaments_df(df):
    """
    Make a dataframe with just the special tournament Jeopardy! episodes
    """
    special_tournaments_df = df[df['notes'] != '-']
    special_tournaments_df = rename_columns(special_tournaments_df)
    special_tournaments_df = make_q_and_a_col(special_tournaments_df)
    # special_tournaments_df = make_clue_difficulty_col(special_tournaments_df)
    return special_tournaments_df


if __name__ == "__main__":
    # Read in the jeopardy.tsv file
    jeopardy_df = read_tsv('../data/master_season1-35.tsv')

    # make dataframes according to special episodes or regular tournaments
    regular_episodes = make_regular_episides_df(jeopardy_df)
    special_tournaments = make_special_tournaments_df(jeopardy_df)

    # Create .csv files of regular_episodes.csv and special_tournaments.csv
    regular_episodes.to_csv("../data/regular_episodes.csv")
    special_tournaments.to_csv("../data/special_tournaments.csv")
