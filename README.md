
# What Is... an Investigation of the Words in Jeopardy!

## Table of Contents
- [The Goal](#The-Goal)
- [Background](#Background)
- [Motivation](#Motivation)
- [Key Terms](#Key-Terms)
- [The Data](#The-Data)
- [EDA](#Exploring-the-Data)
- [Analysis](#Analysis)
- [Conclusion and Futher Analysis](#Conclusion-and-Further-Analysis)
- [Notes, Sources, and Thanks](#Notes,-Sources,-and-Thanks)


## The Goal
**The goal of this analysis is to help *Jeopardy!* home viewers and aspiring contestants study the most common or important subjects that appear in Jeopardy.**

It is straightforward to identify the most common words or categories that have appeared in the show, but that is not enough information for focused study. Instead, I seek to identify the different *meta-categories* that describe groups of similar categories. Within these groupings, I also seek to idenfity which words are most important in order for a contestant to focus their study. 

## Background 
*Jeopardy!* is a trivia gameshow that has been on the air since 1964 (the most recent iteration started in 1984), where three contestants compete against each other -and the clock- by responding to clues. Ae unique feature of *Jeopardy!* is that the host poses the **answer**, and the contestant presses a buzzer to respond to the answer in the form of a **question**, always starting their response with "What is" or "Who is", etc.
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/jeopardy_board.png" alt="gameboard" width="700" height="500"> 
</p>
<p>
<sub>image source:https://en.wikipedia.org/wiki/Jeopardy!<sub>

Each episode of *Jeopardy!* has 61 clues over three rounds, with the following details:

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

## Key Terms

- **Clue**: What I will be calling a category-question-answer combination, which is one instance or observation.
- **J-Category**: The *Jeopardy!* defined category. In the image above, 'EDIBLE RHYME TIME', 'BOOKS IN GERMAN', etc are the J-Categories of one round
**Answer**: The clue read by the host and shown on the screen
**Question:**: The response to the clue, and it must be in the form of a question like "What is..."
**Meta-category**: An overarching topic that can describe each clue's context. For example, the J-Category "EDIBLE RHYME TIME" seen above might belong to a metacategory "Literature" or "Food". In data-science, we can also think of a meta-category as a *latent topic*

## Motivation

The J-Categories in each episode fall under greater themes like pop-culture, history, and literature. But, with 13 J-Categories per episode, and over 8,000 episodes, there can't be *that* many truly unique topics! 
 
Previous contestants and avid fans like myself have intuitions about which themes or topics you should study up if you want to participate at home, or to be a real contestant on the show. For example many *Jeopardy!* fans will say that you need to know about the US presidents, world geography, state capitals, and names of celebrities. 



## The Data
The original dataset is a .txt file, downloaded from [kaggle](https://www.kaggle.com/prondeau/350000-jeopardy-questions) and has 349,641 rows and 9 columns. Each row contains the information pertaining to a single clue over 35 seasons of *Jeopardy!*, from 9/10/1984 to 7/26/2019. They dataset contains information in the 'notes' about whether it was an special tournament or a regular episode. 

#### The original Dataset, read in to a Pandas DataFrame:


|    |   round |   value | daily_double   | category       | comments   | answer                                                    | question     | air_date   | notes   | question_and_answer                                                    | clue_difficulty   |
|---:|--------:|--------:|:---------------|:---------------|:-----------|:----------------------------------------------------------|:-------------|:-----------|:--------|:-----------------------------------------------------------------------|:------------------|
|  0 |       1 |     100 | no             | LAKES & RIVERS | -          | River mentioned most often in the Bible                   | the Jordan   | 1984-09-10 | -       | the Jordan River mentioned most often in the Bible                     | easy              |
|  1 |       1 |     200 | no             | LAKES & RIVERS | -          | Scottish word for lake                                    | loch         | 1984-09-10 | -       | loch Scottish word for lake                                            | easy              |
|  2 |       1 |     400 | no             | LAKES & RIVERS | -          | American river only 33 miles shorter than the Mississippi | the Missouri | 1984-09-10 | -       | the Missouri American river only 33 miles shorter than the Mississippi | easy              |

#### The Updated Dataset
- Each row can be considered a *clue* instance
- I used just the "regular episodes" (not a special tournament or championship), so the 'notes' column was removed
- The column for "comments" is removed, which were additional hints that the host gave to contestants after reading a category
- The "category" column was renamed to "J-Category"
- The 'Question' and 'Answer' columns combined in a 'Question and Answer' column for convenient analysis, as the important words I want to capture within a clue could appear in the answer or question
- A new column called "Question Difficulty"  defines a question as easy, average, or hard<sup>1</sup>
- Punctuation has been removed from the "Question", "Answer", "Question and Answer" columns, but not from the J-category column since there were more than 10 categories over time that had a puncuation mark as the category
- I left the capitalization as-is, and did not remove stop words so I could adjust these settings while tuning my model
-

|    |   Round |   Value | Daily Double   | J-Category    | Answer                                                    | Question     | Air Date   | Question and Answer                                                    | Clue Difficulty   |
|---:|--------:|--------:|:---------------|:--------------|:----------------------------------------------------------|:-------------|:-----------|:-----------------------------------------------------------------------|:------------------|
|  0 |       1 |     100 | no             | LAKES & RIVERS | River mentioned most often in the Bible                   | the Jordan   | 1984-09-10 | the Jordan River mentioned most often in the Bible                     | easy              |
|  1 |       1 |     200 | no             | LAKES & RIVERS | Scottish word for lake                                    | loch         | 1984-09-10 | loch Scottish word for lake                                            | easy              |
|  2 |       1 |     400 | no             | LAKES & RIVERS | American river only 33 miles shorter than the Mississippi | the Missouri | 1984-09-10 | the Missouri American river only 33 miles shorter than the Mississippi | easy   

## Exploring the Data
<p align="center">

### Most Common Words in all J-Categories
<p>
<p align="center">
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/J-Category_wordcloud.png" alt="categories" width="500" height="500">
</p>
These words don't look very helpful for studying...
<p align="center">

### Top 10 Most Common J-Categories 
</p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/top_10_categories.png" alt="length_of_answers" width="600" height="400">
</p>
Look at all that History! But I wish I knew specifically what topics...
</p>

<p align="center">

### Most Common Words in Questions and Answers
<p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/Question%20and%20Answer_wordcloud.png" alt="categories" width="500" height="500">
</p>
<p>
Taking a deeper dive into the words of each clue, (questions and answers combined), these words also don't look very informative either... 
<p align="center">


## Analysis 
I used TF-IDF vectorizer (Term Frequency * Inverse Document Frequency) to vectorize the documents - in other words, I turned the raw questions and answers text into a matrix of the numerical TF-IDF features. This tak
The most important words and these metacategories!?

- **Handling Stopwords** Stopwords are a set of words that do not add significant value to a text, and are often so commonly used that removing them let's an analysis focus on the more important and differentiating words.
- Common stopwords are "the", "or", "and", and I added more stopwords including "one", "word", and "name", 'war', 'film', john', 'state', 'country', 'us', 'new' because they appeared so often and are not specific enough to help someone study.

- **Deciding on the Number of Topics** : I chose to use my domain knowledge to choose the number of topics, as well as testing how well my model ran with different topics, judging against the reconstruction error of the matrix. 
- **Top Words per *meta-category***: I chose top 10 words per category because that seemed like a manageable start for someone studying


## Conclusion and Further Analysis
- What this tells me about the topics to study
- What featurizations do I need to adjust (n-grams, capitalization, n-topics)
- next - use a different vectorizer, like word2vec 



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
A special thank you to Galvanize instructors and residents Kayla, Chris, Rosie, Martha, Alex, and Jenny, and to my scrum group of fellow NLP investigators: Pedro, Ian, Jeff and Devon

This project is dedicated to the late Alex Trebec, the beloved host of *Jeopardy!* for 37 seasons and my friend Laura whose love of Jeopardy inspired this investigation. 


