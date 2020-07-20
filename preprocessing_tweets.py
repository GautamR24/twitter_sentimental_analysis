

from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# this module will preprocess the tweets in three steps
# first it will remove punctuations
# second it will remove the stop words
# third it will normalize the tweets

def text_processing(tweet):
    # this function will remove all the punctuations
    def form(tweet):
        tweet_blob = TextBlob(tweet)
        return ''.join(tweet_blob.words)
    tweet_without_punctuation = form(tweet)
    # this function will remove all the stop words like is,are,have etc
    # because these words does not tell about the sentiments much
    def no_stop_words(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_stop_words(tweet_without_punctuation)
    
    # this function will normalize the text
    # we can have same word written in different tense so normalize it to
    # single word like clap,claps,clapped means same thing but it is written in
    # different tense so we normalize it to one word clap
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)


