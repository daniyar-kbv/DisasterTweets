#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 01:23:25 2022

@author: tanvirsinghahuja
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import string
from wordcloud import WordCloud

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from wordcloud import STOPWORDS
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

train = pd.read_csv('/Users/tanvirsinghahuja/Desktop/disaster_tweets/train.csv')
test = pd.read_csv('/Users/tanvirsinghahuja/Desktop/disaster_tweets/test.csv')
submission = pd.read_csv('/Users/tanvirsinghahuja/Desktop/disaster_tweets/sample_submission.csv')

train.head(5)
train.tail()

print("Sample disaster tweet: ",train[train["target"] == 1]["text"].values[123])
print("Sample non-disaster tweet: ", train[train["target"] == 0]["text"].values[3])

train[["id","target"]].groupby(["target"]).count()

graph = sns.countplot(train.target)
graph.set(xlabel="Is it a real disaster?", ylabel = "Count")

test.head(5)
test.tail()



# define functions for removing noises from text
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
  
def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text 

def remove_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

stop = set(stopwords.words('english'))
def remove_stop_words(text):
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop] 
    filtered_tweet = ' '.join(filtered_sentence)
    return filtered_tweet

def clean(text):
  remove_numbers(text)
  remove_url(text)
  remove_html(text)
  remove_emoji(text)
  remove_punct(text)
  remove_stop_words(text)
  
# apply cleaning function on the text
train_cleaned = train
test_cleaned = test
train_cleaned['text']=train_cleaned['text'].apply(lambda x : remove_stop_words(x))
train_cleaned['text']=train_cleaned['text'].apply(lambda x : remove_punct(x))
train_cleaned['text']=train_cleaned['text'].apply(lambda x : remove_numbers(x))
train_cleaned['text']=train_cleaned['text'].apply(lambda x : remove_html(x))
train_cleaned['text']=train_cleaned['text'].apply(lambda x : remove_emoji(x))

test_cleaned['text']=test_cleaned['text'].apply(lambda x : remove_stop_words(x))
test_cleaned['text']=test_cleaned['text'].apply(lambda x : remove_punct(x))
test_cleaned['text']=test_cleaned['text'].apply(lambda x : remove_numbers(x))
test_cleaned['text']=test_cleaned['text'].apply(lambda x : remove_html(x))
test_cleaned['text']=test_cleaned['text'].apply(lambda x : remove_emoji(x))

disaster = ' '.join([text for text in train_cleaned['text'][train_cleaned['target']==1]])
wordcloud = WordCloud(max_font_size=110).generate(disaster)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()

nondisaster = ' '.join([text for text in train_cleaned['text'][train_cleaned['target']==0]])
wordcloud = WordCloud(max_font_size=110).generate(nondisaster)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=train[missing_cols].isnull().sum().index, y=train[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x=test[missing_cols].isnull().sum().index, y=test[missing_cols].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Training Set', fontsize=13)
axes[1].set_title('Test Set', fontsize=13)

plt.show()

for t in [train, test]:
    for col in ['keyword', 'location']:
        t[col] = t[col].fillna(f'no_{col}')
        
# word_count
train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))
test['word_count'] = test['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
train['unique_word_count'] = train['text'].apply(lambda x: len(set(str(x).split())))
test['unique_word_count'] = test['text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
train['stop_word_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
test['stop_word_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# url_count
train['url_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
test['url_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

# mean_word_length
train['mean_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
train['char_count'] = train['text'].apply(lambda x: len(str(x)))
test['char_count'] = test['text'].apply(lambda x: len(str(x)))

# punctuation_count
train['punctuation_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test['punctuation_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# hashtag_count
train['hashtag_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
test['hashtag_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# mention_count
train['mention_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
test['mention_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length',
                'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']
DISASTER_TWEETS = train['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(train.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[i][0], color='green')
    sns.distplot(train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0], color='red')

    sns.distplot(train[feature], label='Training', ax=axes[i][1])
    sns.distplot(test[feature], label='Test', ax=axes[i][1])
    
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()
    
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

plt.show()

random_state_split = 20
learning_rate = 6e-6
valid = 0.2
epochs_num = 3
batch_size_num = 16

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub


from bert import tokenization
import os
url = 'https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py'
os.system(f"""wget -c --read-timeout=5 --tries=0 "{url}""")
          
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    
    
    out = Dense(1, activation='sigmoid')(clf_output)
   

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values

model_BERT = build_model(bert_layer, max_len=160)
model_BERT.summary()

checkpoint = ModelCheckpoint('model_BERT.h5', monitor='val_loss', save_best_only=True)

train_history = model_BERT.fit(
    train_input, train_labels,
    validation_split = valid,
    epochs = epochs_num, # recomended 3-5 epochs
    callbacks=[checkpoint],
    batch_size = batch_size_num
)



    