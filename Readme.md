# Project: Twitter Sentimental Analysis
In this project we are predicting the sentiments of the tweet based on the words used in the it.
## Libraries used:
1. pandas
2. sklearn
3. nltk
4. textblob
5. re
## Dataset
you can download the dataset from [here](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech )
## Code
There are two files present `sentimental_analysis.py` and `preprocessing_tweets.py`. So while you are executing the `sentimental_analysis.py` you have to include `preprocessing_tweets.py` file also.
## About files
1. `sentimental_analysis.py` 
    * first importing the dataset
    * on observing the dataset we came to know the there are two main things `tweets` and `labels`
    * so we extract both columns for training and testing.
    * now we perform the preprocessing on the tweets using the `preprocessing_tweets.py` module.
    * After preprocessing we simply split the dataset into testing and training dataset using `train_test_split`.
    * Now, we will create a pipeline object which will contain three methods `CounterVectorizer()`, `TfidfTransformer()`, `MultinomialNB()`.
    * So the funtion of each of the above method is explaines below:
       * `CounterVectorizer()` will create a bag-of-words for every tweet. For example the tweet is `Machine learning is good`. So, first it will create a `word_index` like `{'Machine':1,'learning':1,'is':1,'good':1}` then it will create a table to store the information about the occurance of different words in the tweets.
       * `TfidfTransformer()` will calculate the weightage of a word based on its occurance in each tweet.
    * Then we fit this **pipeline object** to the training dataset.
    * At last we test the model on the test dataset.
  2. `preprocessing_tweet.py`
      * In this the main function is `text_processing()` under which there are three function `form()`,`no_stop_words()`,`normalization()`.
      * The work of each of the above funtion is as follows:
         * `from()` will take each tweet as input and remove all the punctuations as it doesn't tell about the sentiments.
         * `no_stop_words()` will remove the words like is, are, have as these also don't tell about the sentiments.
         * `normalization()` will normalize same word present in different tense/ways to single word like enjoy, enjoying, enjoyed all mean the same, so , it will replaced with single word enjoy. 
         * A last the tweet is completely processed.
## Methodology
1. import the required libraries.
2. import the datasets.
3. extact the `label` and `tweets` columns.
4. do the preprocessing of the `tweets` in train and test dataset.
5. split the dataset into testing and training dataset using `train_test_split`
6. create the `pipeline object` containing the three functions `CountVectorier()`,`TfidfTransformer()`,`MultinomialNB()`.
7. fit `pipeline object` to the training dataset.
8. predict output for the test dataset using this `pipeline object`.
9. At last print the accuracy.

## Result
Using Naive Bayes got the accuracy of  **93%**. 


                                                   




