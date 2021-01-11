
# What are... The Most Important Topics to Study for "Jeopardy!"

## Table of Contents
- [The Goal](#The-Goal)
- [Background](#Background)
- [Motivation](#Motivation)
- [Key Terms](#Key-Terms)
- [The Data](#The-Data)
- [EDA](#Exploring-the-Data)
- [Analysis](#Analysis)
- [Conclusion and Recommendation](#Conclusion-and-Recommendation)
- [Next Steps](#Next-Steps)
- [Notes, Sources, and Thanks](#Notes,-Sources,-and-Thanks)


## The Goal
**The goal of this analysis is to help "Jeopardy!" home viewers and aspiring contestants study the most common or important subjects that appear in "Jeopardy!".**


## Background 
"Jeopardy!" is a trivia gameshow that has been on the air since 1964 (the most recent iteration started in 1984), where three contestants compete against each other -and the clock- by responding to clues. A unique Yfeature of "Jeopardy!" is that the host poses the **answer**, and the contestant presses a buzzer to respond to the answer in the form of a **question**, always starting their response with "What is..." or "Who is...", etc.
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/jeopardy_board.png" alt="gameboard" width="700" height="500"> <sub>figure1</sub>
</p>
<p>
<sub>image source:https://en.wikipedia.org/wiki/Jeopardy!<sub>

Each episode of "Jeopardy!" has 61 clues over three rounds, with the following details:

- Round 1: Jeopardy!
    - six "Jeopardy!"-defined categories (referred to as **J-Categories** from here on) with 5 clues each
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

- **Clue**: What I will be calling a question-answer combination. A single clue instance can be considered as a text document 
- **J-Category**: The "Jeopardy!" defined category. In the image above, 'EDIBLE RHYME TIME', 'BOOKS IN GERMAN', etc are the J-Categories of one round
- **Answer**: A description of  is read by the host and shown on the screen. 
- **Question**: The response to the answer given by the host, and it must be in the form of a question like "What is..."
- **Meta-category**: An overarching topic that can describe each clue's context, also referred to as a **hidden theme**. For example, the J-Category "EDIBLE RHYME TIME" seen above might belong to potential meta-categories "Literature" and "Food". In data-science, we can also think of a meta-category as a **latent topic**

## Motivation

The J-Categories in each episode fall under greater themes like pop-culture, history, and literature. But, with 13 J-Categories per episode, and over 8,000 episodes, there can't be *that* many truly unique topics! 

It is straightforward to identify the most common words or categories that have appeared in the show, but that is not enough information for focused study. Instead, I seek to identify the different *meta-categories* that describe groups of similar categories. Within these groupings, I also seek to idenfity which words are most important in order for a contestant to focus their study. 


## The Data
The original dataset is a .tsv file, downloaded from [Kaggle](https://www.kaggle.com/prondeau/350000-jeopardy-questions) and has 349,641 rows and 9 columns. Each row contains the information pertaining to a single clue per episode over 35 seasons of "Jeopardy!", from 9/10/1984 to 7/26/2019. The dataset contains information in the 'notes' about whether it was an special tournament or a regular episode.

#### The Original, Raw Dataset
|Round|Value|Daily Double|J-Category    |Answer                                                   |Question    |Air Date|
|-----|-----|------------|--------------|---------------------------------------------------------|------------|--------|
|1    |100  |no          |LAKES & RIVERS|River mentioned most often in the Bible                  |the Jordan  |9/10/84 |
|1    |200  |no          |LAKES & RIVERS|Scottish word for lake                                   |loch        |9/10/84 |
|1    |400  |no          |LAKES & RIVERS|American river only 33 miles shorter than the Mississippi|the Missouri|9/10/84 |

<sub>table1</sub>

#### The Dataset Used for This Project
The episodes considered in this analysis only consider the episode that are not special tournamnets (I refer to these to as "Regular Episodes"). There are 278,730 total episodes included in this dataset.

|   Round |   Value | Daily Double   | J-Category    | Answer                                                    | Question     | Air Date   | Question and Answer                                                    | Clue Difficulty   |
|---:|--------:|--------:|:---------------|:--------------|:----------------------------------------------------------|:-------------|:-----------|:-----------------------------------------------------------------------|:------------------|
|       1 |     100 | no             | LAKES & RIVERS | River mentioned most often in the Bible                   | the Jordan   | 1984-09-10 | the Jordan River mentioned most often in the Bible                     | easy              |
|       1 |     200 | no             | LAKES & RIVERS | Scottish word for lake                                    | loch         | 1984-09-10 | loch Scottish word for lake                                            | easy              |
|       1 |     400 | no             | LAKES & RIVERS | American river only 33 miles shorter than the Mississippi | the Missouri | 1984-09-10 | the Missouri American river only 33 miles shorter than the Mississippi | easy  
<sub>table2</sub> 

In the above table:
- Each row can be considered a *clue* instance
- The only shows the regular episodes (not a special tournament or championship), so the "notes" column was removed
- The column for "comments" is removed, which were additional hints that the host gave to contestants after reading a category
- The "Category" column was renamed to "J-Category"
- The "Question" and "Answer" columns combined in a "Question and Answer" column for convenient analysis, as the important words I want to capture within a clue could appear in the answer or question
- A new column called "Question Difficulty" is added, that defines a question as easy, average, or hard<sup>1</sup>
- Punctuation has been removed from the "Question", "Answer", "Question and Answer" columns, but not from the J-category column since there were more than 10 categories over time that had a single puncuation mark as the category
- Capitalization is left as-is for now, and did not remove stopwords so I could adjust these settings while tuning my model. 

## Exploring the Data
<p align="center">

### Most Common Words in all J-Categories
<p>
<p align="center">
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/J-Category_wordcloud.png" alt="categories" width="500" height="500"><sub>figure2</sub>
</p>
I only removed basic stopwords from NLTK's stopwords set for this visual. Even after that, these words don't look very helpful for focused studying...
<p align="center">

### Top 10 Most Common J-Categories 
</p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/top_10_categories_blue.png" alt="length_of_answers" width="650" height="500"> <sub>figure3</sub>
</p>
Look at all that History! But I wish I knew specifically what History I should study for the most impact on my "Jeopardy!" skills...
</p>

<p align="center">

### Most Common Words in Questions and Answers
<p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/eda_images/Question%20and%20Answer_wordcloud.png" alt="categories" width="500" height="500"><sub>figure4</sub>
</p>
<p>
Taking a deeper dive into the words within each clue, (questions and answers combined). Even after removing NLTK's basic stopwords, these words also don't look super helpful for targeted study either... 
<p align="center">


## Analysis 

#### Workflow
<p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/workflow.png", alt="workflow" width=500 height=500>
</p>

### Model Selection 

* I used a tf-idf (Term Frequency * Inverse Document Frequency) to vectorize the text from each clue. In other words, I turned the raw text from the "Jeopardy!" questions and answers into a matrix whose entries are the numerical TF-IDF features of each word in the text. 
* I then used Non-Negative Matrix Factorization (NMF) to create clusters of words, where each cluster can be thought of as a *meta-category* or latent topic, which is one of the goals of this analysis.

NMF is a *soft clustering* model, which in this context means that any clue could belong to multiple clusters. This is advantageous because the nature of "Jeopardy!" is that clues often touch on multiple topics. This is also advantageous for studying purposes: if a word from a single document appears in multiple clusters, you get more "bang for your buck" by studying the information around that word. This makes sense for my goal, because as said above with the "EDIBLE RHYME TIME"  example, that clue could be a part of mutiple clusters such as Literature or Food. Another benefit of using NMF for the topic modeling is that the loading or weights of each word in a cluster is positive, so their importance is more easily interpreted. 

### Model Settings (Hyperparameters)

- **Number of Topics** : I compared the reconstruction error of the matrices formed by the Non-Negative Matrix Factorization to the number of topics. In essence, this is a metric that measures how different the values in the reconstructed matrix of tf-idf value are from the original matrix. 
<p>
<p align="center">
<img src="https://github.com/JCurley10/Jeopardy_nlp/blob/main/images/results_images/reconerr_vs_k.png" alt="recon_vs_k" width="700" height="550">
</p>
* Since my goal is to help a user focus their attention on important topics, I chose to only consider between 7 and 15 topics or clusters. In the end, the trend of the reconstruction matrix was close to linear, as seen above and not very meaningful: the greater the number of clusters, the lower the reconstruction error. This means that I could very easily decide on 100 clusters beause it had a low reconstruction error, but would lose the cohesiveness within the clusters, or have too many topics to focus on! 
* After checking the cohesiveness of the topics manually, I found that 13 clusters produced the most meaningful groupings of words! 
* An added coincidence here is that each episode of "Jeopardy!" has 13 categories, so 13 was my lucky number. 

- **Top Words per *meta-category***: I chose top 10 words per category because it is a manageable start for someone planning on studying for "Jeopardy!"
- **Handling Stopwords, Tokenization, and N-grams** : Stopwords are a set of words that do not add significant value to a text, and are often so commonly used that removing them let's an analysis focus on the more important and differentiating words.
    - Common stopwords are "the", "or", "and", which were already in my original stopwords set taken from NLTK. I added more stopwords including "one", "word", and "name", "war", "film", "state', "country", "us", and "new" because they appeared so often and are not specific enough to help someone study specific words.
    - I chose to tokenize the words using NLTK's WordNetLemmatizer, although it still produced some messy words I had to handle within my stopwords set
    - I did set the option of including n-grams = 2, to allow words like "North Dakota" to appear in the analysis, but 2-grams didn't show up as a top 10 words per cluster. 

### Visual
Below are wordclouds that show the top 15 words that appear within each of the 13 clusters. The size of the word relates to the weight of the word as it appears in the Words vs. Hidden Topics matrix (example shown in figure)

| TOPIC | TOPIC | TOPIC |
|-|-|-|
| <img src="" alt="categories" width="300" height="275"> | <img src="https://github.com/JCurley10/" alt="categories" width="300" height="275"> | <img src="" alt="categories" width="300" height="275"> |

| TOPIC | TOPIC | TOPIC |
|-|-|-|
|<img src="" alt="categories" width="300" height="275"> |<img src="" alt="categories" width="300" height="275"> |<img src="" alt="categories" width="300" height="275"> |

| TOPIC | TOPIC | TOPIC |
|-|-|-|
| <img src="" alt="categories" width="300" height="275"> | <img src="" alt="categories" width="300" height="275"> | <img src="" alt="categories" width="300" height="275"> | 

| TOPIC | TOPIC | TOPIC |
|-|-|-|
| <img src="" alt="categories" width="300" height="275"> | <img src="" alt="categories" width="300" height="275"> | <img src="" alt="categories" width="300" height="275"> |

|TOPIC|
|-|
|<img src="" alt="categories" width="300" height="275"> | 

<sub>figure5</sub>

## Conclusion and Recommendation IN PROGRESS

#### If you are studying to be a "Jeopardy!" contestant, you should focus your attention on the following:

| **Meta-Category**| TOPIC | TOPIC |  |  |  |  |  |   |  |  |
|-|-|-|-|-|-|-|-|-|-|-|
| **Specific Words/Topics**| |  |  |  |  | |  | |  |

As I expected, Geography, Literature, and Pop Culture are very important. Busines and Industry, which is one of the most common categories as seen above, was also a clear cluster. I am interested in why Science and History terms are not so clear in these clusters, given they are such common categories. Maybe, there are *just too many* common words for History and Science that they weren't included due to the nature of TF-IDF vectorization. Or, it could be because NMF is a soft-clustering model, the common words in History and Science categories can be found within other clusters. 
<sub>table3</sub>

## Next Steps
- Keep the model updated to account for newer episodes
- Implement Word2Vec as a suggestion, which can do a better job of learning word embeddings by taking into account surrounding words 
- Use a latent Dirichlet allocation (LDA) model to perform a similar analysis and compare with this NMF model 
- Use these clusters as a feature in a supervised learning model for classifying questions as easy, medium or hard

### Notes, Sources and Thanks

<sup>1</sup> While note used in this particular iteration of the project, I decided to classify a clue's difficulty using an analysis done by a fellow data scientist named Colin Pollock, found [here](https://medium.com/@pollockcolin/jeopardy-question-difficulty-1bba47770bc6). I used his "Percent Correct by Round and Value" chart to decide what makes a clue easy, average, or hard. The average percent correct was around 50% according to this graph, so I decided an average success under 50% was classified as "hard", between 50% and 60% "average", and over 60% "easy". The following types of clues were classified as such:

- easy clues: value less than or equal to $600, and not a daily double, in either category 1 or 2
- average clues: a daily double in category 1, or a value equal to $800 in either category
- hard clues: a daily double in category 2, a value equal to $1,000 in either category 1 or 2, a value greater than or equal to $1600, or Final Jeopardy

This varied slightly than my own assumptions, which are:
- easy clues: Less than $800 in either round, and not a dailty double
- average clues: over $800 in round 1, $1200 in round 2, or a daily double in round 1
- hard clues: over $1200 in round 2, a daily double in round 2, or final Jeopardy

**Thanks**<p>
- A special thank you to Galvanize instructors and residents Kayla, Chris, Rosie, Martha, Alex, and Jenny, and to my scrum group of fellow NLP investigators: Pedro, Ian, Jeff and Devon. <p>
- This project is dedicated to the late Alex Trebec, the beloved host of "Jeopardy!" for 37 seasons, and to my friend Laura whose love of "Jeopardy!" inspired this investigation. 
