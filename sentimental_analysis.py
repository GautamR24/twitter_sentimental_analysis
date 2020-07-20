
import preprocessing_tweets as pt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#importing the test and train dataset files
tweets_file_train = pd.read_csv("train.csv")
tweets_file_test = pd.read_csv("test.csv")

# there are two main columns in the dataset ,ie, label and tweet columns,
# so just using these two columns to train the model then w'll test on the 
# test dataset
train_tweets = tweets_file_train[['label','tweet']]
test_tweets = tweets_file_test[['tweet']]


# adding new column to the table which will contain the processed tweets
# in processing all the punctuations, stop words will be removed and
# we will get a list containing only those words which will help in predicting
# the sentiments
train_tweets['tweet_list'] = train_tweets['tweet'].apply(pt.text_processing)
test_tweets['tweet_list'] = test_tweets['tweet'].apply(pt.text_processing)


# splitting the dataset into testing and training set using train_test_split
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets['label'], test_size=0.2)   


# using pipeline to automate the tasks
# it will first create a bag-of-words using counter vectorization based on the 
# preprocessing module preprocessing_tweets.py
# second, using tf-idf it will calculate the weightage of each word
# third, it will use naive bayes classifier 
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=pt.text_processing)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# fitting into the training dataset
pipeline.fit(msg_train,label_train)

# testing the trained model on test dataset
predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))
print ('\n')
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))
