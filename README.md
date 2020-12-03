
# What Is... an Investigation of the Words in Jeopardy!

## Table of Contents
- [Background and Motivation](#Background-and-Motivation)
- [The Goal](#The-Goal)
- [Key Terms](#Key-Terms)
- [The Data](#The-Data)
- [EDA](#Exploring-the-Data)
- [Analysis](#Analysis)
- [Conclusion and Futher Analysis](#Conclusion-and-Further-Analysis)
- [Notes, Sources, and Thanks](#Notes,-Sources,-and-Thanks)

## Background and Motivation
*Jeopardy!* is a gameshow that has been on the air since 1964 (the most recent iteration started in 1984), where three contestants compete against each other -and the clock- by responding to trivia clues with different values. A unique feature of *Jeopardy!* is that the host poses an **answer** and the contestant who buzzes in first responds in the form of a **question**, always starting their response with "What is" or "Who is", etc.
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/jeopardy_board.png" alt="gameboard" width="700" height="500"> 
</p>
<p>
<sub>image source:https://en.wikipedia.org/wiki/Jeopardy!<sub>

There are three rounds in each episode with the following details:

- Round 1: Jeopardy!
    - six *Jeopardy!*-defined categories (referred to as **J-Categories** from here on) with 5 clues each
    - clue values: $200, $400, $600, $800, $1000
    - 1 Daily Double
- Round 2: Double Jeopardy!
    - six J-Categories with 5 clues each
    - clue values: $400, $800, $1200, $1600, $2000
    - 2 Daily Doubles
- Round 3: Final Jeopardy!
    - one question each contestant answers and wagers how much the win if they respond correctly 

With Daily Doubles and Final Jeopardy, contestants can wager a minimum of $5, their entire winnings so far (known as a "True Daily Double"), or the most expensive clue that round ($1000 or $2000).

The J-Categories in each episode fall under greater themes like pop-culture, history, and literature. But, with 13 J-Categories per episode, and over 8,000 episodes, there can't be *that* many truly unique topics! 
 
Previous contestants and avid fans like myself have intuitions about which themes or topics you should study up if you want to participate at home, or to be a real contestant on the show. For example many *Jeopardy!* fans will say that you need to know the US presidents, world geography, state capitals, and names of celebrities. 

## Key Terms
#### Clue
What I will be calling a category-question-answer combination, which is one instance or observation.
#### J-Category: 
The *Jeopardy!* defined category. In the image above, 'EDIBLE RHYME TIME', 'BOOKS IN GERMAN', etc are the J-Categories of one round
#### Answer: 
The clue read by the host, and shown on the screen
#### Question:
The response to the clue, and it must be in the form of a question like "What is..."
#### meta-category
An overarching topic that can describe each clue's context. For example, the J-Category "EDIBLE RHYME TIME" seen above might belong to a metacategory "Literature" or "Food". In data-science, we can also think of a meta-category as a *latent topic*


## The Goal
**The goal of this project is to help interactive *Jeopardy!* home viewers and aspiring contestants study the most common subjects that appear in Jeopardy.**

It is very straightforward to identify the most common words or J-Categories that have appeared in the show, but that does not help with focused study. Instead, I seek to identify the different *meta-categories* that describe groups of similar J-Categories, and within these groupings, idenfity which words are most important in order for a contestant to focus their study. 


*Jeopardy!*
## The Data
The original dataset is a .txt file, downloaded from [kaggle](https://www.kaggle.com/prondeau/350000-jeopardy-questions) and has 349,641 rows and 9 columns. Each row contains the information pertaining to a single clue over 35 seasons of *Jeopardy!*, from 9/10/1984 to 7/26/2019.

#### The original Dataset, read in to a Pandas DataFrame:


|    |   round |   value | daily_double   | category       | comments   | answer                                                    | question     | air_date   | notes   | question_and_answer                                                    | clue_difficulty   |
|---:|--------:|--------:|:---------------|:---------------|:-----------|:----------------------------------------------------------|:-------------|:-----------|:--------|:-----------------------------------------------------------------------|:------------------|
|  0 |       1 |     100 | no             | LAKES & RIVERS | -          | River mentioned most often in the Bible                   | the Jordan   | 1984-09-10 | -       | the Jordan River mentioned most often in the Bible                     | easy              |
|  1 |       1 |     200 | no             | LAKES & RIVERS | -          | Scottish word for lake                                    | loch         | 1984-09-10 | -       | loch Scottish word for lake                                            | easy              |
|  2 |       1 |     400 | no             | LAKES & RIVERS | -          | American river only 33 miles shorter than the Mississippi | the Missouri | 1984-09-10 | -       | the Missouri American river only 33 miles shorter than the Mississippi | easy              |

#### The Updated Dataset
- We're looking at regular episodes (not a special tournament or championship), so the 'notes' column was removed
- The column for "comments" is removed, which were additional hints that the host gave to contestants after reading a category
- The 'question' and 'answer' columns combined in a 'Question and Answer' column for analysis
- A new column called "question_difficulty"  defines a question as easy, average, or hard<sup>1</sup>
- Punctuation has been removed
- I left the capitalization as-is for all columns so I could adjust this setting in my model
- each row can be considered a *clue* instance

|    |   Round |   Value | Daily Double   | J-Category    | Answer                                                    | Question     | Air Date   | Question and Answer                                                    | clue_difficulty   |
|---:|--------:|--------:|:---------------|:--------------|:----------------------------------------------------------|:-------------|:-----------|:-----------------------------------------------------------------------|:------------------|
|  0 |       1 |     100 | no             | LAKES  RIVERS | River mentioned most often in the Bible                   | the Jordan   | 1984-09-10 | the Jordan River mentioned most often in the Bible                     | easy              |
|  1 |       1 |     200 | no             | LAKES  RIVERS | Scottish word for lake                                    | loch         | 1984-09-10 | loch Scottish word for lake                                            | easy              |
|  2 |       1 |     400 | no             | LAKES  RIVERS | American river only 33 miles shorter than the Mississippi | the Missouri | 1984-09-10 | the Missouri American river only 33 miles shorter than the Mississippi | easy   

## Exploring the Data
<p align="center">

### Most Common Words in all J-Categories
<p>
<p align="center">
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/category_wordcloud.png" alt="categories" width="500" height="500">
</p>

<p align="center">

### Top 10 Most Common J-Categories 
</p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/top_10_categories.png" alt="length_of_answers" width="600" height="400">
</p>
American History, History, World History all appear! 
</p>

<p align="center">

### Most Common Words in Combined Question and Answer
<p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/question_and_answer_wordcloud.png" alt="categories" width="500" height="500">
</p>
<p>

<p align="center">

### Number of Words per Answer, By Difficulty<sup>1</sup>
<p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/Answer%20Word%20Count_counts_bar.png" alt="length_of_answers" width="600" height="400">
</p>
<p>

## Analysis 



### Notes, Sources and Thanks

<sup>1</sup> I decided to classify a clue's difficulty using an analysis done by a fellow data scientist named Colin Pollock, found [here](https://medium.com/@pollockcolin/jeopardy-question-difficulty-1bba47770bc6). I used his "Percent Correct by Round and Value" chart to decide what makes a question easy, average, or hard. The average percent correct was around 50% according to this graph, so I decided An average success under 50% was classified as "hard", between 50% and 60% "average", and over 60% "hard". The the following types of clues were classified as such:

- easy clues: value less than or equal to 600, and not a daily double, in either category 1 or 2
- average clues: a daily double in category 1, or a value equal to 800 in either category
- hard clues: a daily double in category 2, a value equal to $1,000 in either category 1 or 2, a value greater than or equal to $1600, , or Final Jeopardy

This varied slightly than my own assumptions, which are:
- easy clues: Less than $800 in either round, and not a dailty double
- average clues: over $800 in round 1, $1200 in round 2, or a daily double in round 1
- hard clues: over $1200 in round 2, a daily double in round 2, or final jeopardy

**Thanks**<p>
A special thank you to Galvanize instructors and residents Kayla, Chris Martha, Alex, and Jenny, and to my scrum group of fellow NLP investigators: Pedro, Ian, Jeff and Devon

This project is dedicated to the late Alex Trebec, the beloved host of *Jeopardy!* for 37 seasons and my friend Laura whose love of Jeopardy inspired this investigation. 


