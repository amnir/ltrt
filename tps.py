import csv
import json
import re
import string

import numpy as np
from flask_cors import CORS, cross_origin

from gensim import corpora
from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request
from keras_preprocessing import sequence
import hebrew_tokenizer as ht

# Create app
app = Flask(__name__)
CORS(app)

def clean_tweet(tweet):
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'https…', '', tweet)
    # remove hashtags - only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@\S+', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    return tweet

def load_keras_model():
    """Load in the pre-trained model"""
    global model
    model = load_model('lstm_model.hdf5')
    # Required for model to work
    global graph
    graph = tf.get_default_graph()

def build_dict():
    with open('clean_tweets.csv', 'r', encoding='utf-8') as tweets:
        tweets = csv.reader(tweets)
        # next(tweets)  # Skip headers row
        all_words = []
        for tweet in tweets:
            tokenized = ht.tokenize(tweet[0])
            words = [token for grp, token, token_num, (start_index, end_index) in tokenized]
            all_words.append(words)

    global dict
    dict = corpora.Dictionary([word for word in all_words])


# Home page
@app.route("/", methods=['GET', 'POST', 'OPTIONS'])
def home():
    """Home page of app with form"""
    # Create form
    template = render_template('index.html')
    return template


@app.route("/predict", methods=['POST'])
def predict():
    our_tweets_vects = []
    tweet = request.json["tweet"]
    tweet = clean_tweet(tweet).split(' ')
    vector = [dict.token2id[word] for word in tweet if word in dict.values()]
    our_tweets_vects.append(vector)
    our_tweets_np = np.array(our_tweets_vects)
    our_tweets_vects = sequence.pad_sequences(our_tweets_np, maxlen=150)

    with graph.as_default():
        our = model.predict_classes(our_tweets_vects)

    if len(our) != 1:
        raise Exception("Failed to preict")

    ans = {}
    if our[0] == 0:
        ans = {"affiliation": "left"}
    elif our[0] == 1:
        ans = {"affiliation": "right"}
    return json.dumps(ans)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    build_dict()
    load_keras_model()
    # Run app
    app.run(host="127.0.0.1", port=80)
