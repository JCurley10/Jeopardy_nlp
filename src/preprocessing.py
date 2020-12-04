import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF as NMF_sklearn
import string
import re


def read_tsv(filepath):
	"""
	Reads in a tsv file
	"""    
	return pd.read_csv(filepath, sep="\t")


def stringify(df, col):
	"""
	Turns the column (col) into one string,
	to be able to make a wordcloud

	Args:
		df (Pandas dataFrame): The dataframesin use
		col (str): the column name to turn into a string
	Returns:
		a string that connects all the words in the column
	"""        
	return ' '.join(df[col])

def lowercase(df, cols):
"""
	turns the column (col) into one string,
	whose letters are all lowercase
	to be able to make a wordcloud

	Args:
		df (Pandas dataFrame): The dataframe in use
		col (str): the column name to turn into a string
	Returns:
		a dataframe with all lowercase text
		in the given column lowercase
	"""   
	for col in cols: 
		df[col] = df[col].str.lower()
	return df


def remove_punc(df, cols):
	"""
	remove punctuation from a column

	Args:
		df (Pandas dataFrame): The dataframe in use
		col (str): the column name to turn into a string
	Returns:
		Pandas DataFrame: with removed punctuation
	"""    
	for col in cols:
		p = re.compile(r'[^\w\s]+')
		df[col] = [p.sub('', x) for x in df[col].tolist()]
	return df


def tokenize_by_col(df, cols):
	"""
	Tokenizes all the words in a stringified column
	Args:
		df (Pandas dataFrame): The dataframe in use
		col (str): the column name to turn into a string
	Returns:
		a dataframe with each column containing 
		tokenized words
	"""        
	for col in cols:
		df[col] = df[col].apply(word_tokenize)
	return df



def tokenize(text):
	"""[summary]

	Args:
		text (string): a string to be tokenized

	Returns:
		a list of strings: tokenized words 
	"""	
	lemmatizer = WordNetLemmatizer()
	# stemmer = SnowballStemmer('english')
	word_list = word_tokenize(text)

	lemmatized_wrds = [lemmatizer.lemmatize(w) for w in word_list]
	# stemmed_wrds = [stemmer.stem(w) for w in lemmatized_wrds]
	# return stemmed_wrds
	return lemmatized_wrds


def make_stopwords(col):

	if col == 'notes':
		remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'tournament', 'week', 'reunion', 'ultimate'}
		stopwords_set = (set(stopwords.words('english'))).union(remove_words)
	else:
		remove_words = {'man', 'men', 'woman', 'person', 'he', 'she', 'they', 'people', 'with', 
						'also', '000', 'girl', 'boy', 'know', 'mean', 'name', 'word', 'use', 
						'big', 'small', 'one', 'iii', 'two', 'three', 'i', 'ii', 'said', 'say',
						'know', 'known', 'name', 'say', 'good', 'bad', 'group', 'sometime',
						'always', 'never', 'd', 's', 'abov', 'alway', 'ani', 'becaus', 'befor', 
						'could', 'doe', 'dure', 'ha', 'might', 'must', "n't", 'need', 'onc', 'onli', 
						'ourselv', 'peopl', 'sha', 'sometim', 'themselv', 'veri', 'wa', 'whi', 'wo', 'would', 'yourselv' 
						'word', 'named', 'part', 'often', 'meaning', 'called', 'call', 'name', 'like', 'say', 'says'
						'mean', 'means', 'contain', 'contains', 'someone', 'some', 'two', 'three', 'four', 'five'
						'day', 'night', 'great', 'became', 'day', 'around', 'sometimes', 'used', 'use', 'made', 'country'
						'term', '1st', 'got', 'time', 'times', 'tell', 'tells', 'may', 'war', 'film', 'de', 'john', 'state', 'new', 
						'title', 'city', 'type', 'first', 'go', 'years', 'last', 'make', 'country', 'us', 'new', 'book', 'president', 
						'world', 'wrote', 'd', "\'ll", "\'re", "s", "ve", "u", "river", "come", "get", 'novel', 'largest', 'old', 'played', 
						'islands', 'whose', 'comes', 'age', 'thats', 'theyre', 'life', '2word', '_', '1', '2', '3', '4', '5', '20', '10', '30', 'every'}
		stopwords_set = (set(stopwords.words('english'))).union(remove_words)
	return list(stopwords_set)


def remove_stopwords(df, col, stopwords):
	"""
	remove stopwords from a set
	Args:
		df (Pandas DataFrame)
		cols (list of str): list of columns to be cleaned, as strings
	Returns:
		Pandas Dataframe: the original dataframe with the column without stopwords
	"""
	
	df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
	return df


# This function doesn't need anything from the above
def clean_text(df, col):
	'''
	"""    
	taking in a column from a dataframe,
	return a list of tokenized words, stripped of stopwords 
	Args
			df (Pandas DataFrame)
			cols (list of str): list of columns to be cleaned, as strings
	Returns:
		[list of str]: a list of strings
			to be used for an EDA wordcloud from a given column
	"""
	'''
	text = ' '.join(df[col])
	tokens = word_tokenize(text)
	# converts the tokens to lower case
	tokens = [w.lower() for w in tokens]
	# remove punctuation from each word
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]

	words = [word for word in stripped if word.isalpha()]
	
	# filter out stop words
	if col == 'notes':
		#TODO: add another set of stopwords for the notes
		remove_words = {'final', 'quarterfinal', 'game', 'jeopardy!', 'semifinal', 'round', 'He', 'She', 'They', 'This' 'tournament', 'week', 'reunion', 'ultimate', 'night', 'jeopardy', 'night', 'games'}
		stopwords_set = (set(stopwords.words('english'))).union(remove_words)
	else:
		stopwords_set = set(stopwords.words('english'))
	words = [w for w in words if not w in stopwords_set]
	return words


def make_q_and_a_col(df):
	"""
	Makes a column that concatenates the strings
	from the question and answer columns

	Args:
		df (Pandas DataFrame): 
	Returns:
		Pandas DataFrame with an additional column
	"""    
	df['question_and_answer'] = df["question"] + ' ' + df['answer']
	return df

def make_clue_difficulty_col(df, viewer_assumptions = False):
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
		conditions = [((df['daily_double']=='no') & (df['value']<= 800)), #easy
					((df['daily_double']=='no') & (df['round']== 1) & (df['value'] >= 800)), #average
					((df['daily_double']=='no') & (df['value'] == 1200)) #average
					((df['daily_double']=='no') & (df['round']== 2) & (df['value'] >= 1600)), #hard
					((df['daily_double']== 'yes') & (df['round'] == 1)), #average
					((df['daily_double']== 'yes') & (df['round'] == 2)), #hard
					(df['round'] == 3)] # final jeopardy, hard 

		difficulties = ['easy', 'average', 'average', 'hard', 'average', 'hard', 'hard']
		
	else:
		conditions = [((df['value']<=600) & (df['daily_double']=='no')), #easy
				((df['daily_double']=='no') & ((df['value']==800) | (df['value']==1200))), #average
				((df['daily_double']== 'yes') & (df['round'] == 1)), #average
				((df['daily_double']=='no') & ((df['value']==1000) | (df['value']>=1600))), #hard
				((df['daily_double']== 'yes') & (df['round'] == 2)), #hard
				(df['round'] == 3)] # final jeopardy, hard 
	


		difficulties = ['easy', 'average', 'average', 'hard', 'hard', 'hard']

	df['clue_difficulty'] = np.select(conditions, difficulties)
	return df

def update_df_columns(df):
	"""
	update the dataframe with the columns 
	made from the functions make_q_and_a_col 
	and male_clue_difficulty_col

	Args:
		df (Pandas DataFrame): 
	Returns:
		df (Pandas DataFrame) udpated with new columns
	"""    
	df_new = make_q_and_a_col(df)
	df_new = make_clue_difficulty_col(df_new)
	return df_new


#TODO: write docstring
def make_sub_df(df, fraction = .05, state = 123):
	"""
	make a subsample of a dataframe for
	modeling purposes

	Args:
		df ([type]): [description]
		fraction (float, optional): [description]. Defaults to .05.
		state (int, optional): [description]. Defaults to 123.

	Returns:
		Pandas Dataframe: A fraction of the Pandas DataFrame 
	"""
	return df.sample(frac = fraction, axis = 0, random_state = state)


if __name__ == "__main__":
	jeopardy_df = read_tsv('../data/master_season1-35.tsv')

	jeopardy_df = remove_punc(jeopardy_df, ['question', 'answer']) #don't remove punctuation from category since there are categories with only punctuation 
	jeopardy_df = update_df_columns(jeopardy_df)
	jeopardy_df = jeopardy_df.rename(columns={"round": "Round", "value": "Value", "category": "J-Category", "answer": "Answer", 'question':
										   'Question', 'question_and_answer': 'Question and Answer', 'daily_double': 'Daily Double', 'air_date': 'Air Date', 'clue_difficulty': 'Clue Difficulty'})

	regular_episodes = jeopardy_df[jeopardy_df['notes']=='-']
	special_tournaments = jeopardy_df.drop(regular_episodes.index)
	regular_episodes = regular_episodes.drop(['notes', 'comments'], axis =1)
	regular_episodes_sub = make_sub_df(regular_episodes)

	regular_episodes.to_csv('../data/jeopardy_regular_episodes.csv')

	





