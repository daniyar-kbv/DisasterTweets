#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:35:30 2022

@author: harshwardhanbabel
"""

import numpy as np
import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df = train_df[['id','text','target']]

train_df.isnull().values.any()

import re

'''symbols = set()
for text in train_df.text.tolist():
    syms = re.findall(r'\W', text)
    for sym in syms:
        symbols.add(sym) # check for non char or non numerical values
        
symbols
'''
def remove_symbol(x):
    return re.sub(r'\W', ' ', x)

train_df.text = train_df.text.apply(remove_symbol)
symbols = set()
for text in train_df.text.tolist():
    syms = re.findall(r'\W', text)
    for sym in syms:
        symbols.add(sym) # check for non char or non numerical values
train_df.text

train_df.target.value_counts()

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()          
bow.fit(train_df.text)               
X_bow = bow.transform(train_df.text) 
X_bow.toarray()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(train_df.text) 
X_tfidf.toarray()

y = train_df.target
y


'''
X = train_df.drop(['id','target'], axis = 1)
X
'''

from imblearn.over_sampling import SMOTE

oversampling = SMOTE(sampling_strategy='auto')
X, y = oversampling.fit_resample(X_tfidf, y) 
y.value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X,
                                                     y,
                                                     test_size=0.25,
                                                     random_state=1)
n_samples = X_train.shape[0]
n_featuers = X_train.shape[1]


# Make the NN -----------------------------------------------------------------

# Importing the Keras libraries and packages
from keras.layers import Dense
from keras.models import Sequential

#from tensorflow.keras.layers import Dense

# define and initialize the model
my_classifier = Sequential()

# Adding the input layer AND the first hidden layer (Pay attention to this)
my_classifier.add(Dense(units = 20, kernel_initializer = 'uniform',
                        activation = 'tanh', input_dim = n_featuers))

# Adding the second hidden layer
my_classifier.add(Dense(units = 8, kernel_initializer = 'uniform',
                                                    activation = 'sigmoid'))

# Adding the last (output) layer
my_classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                        activation = 'sigmoid'))

# Compiling the ANN
my_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

#my_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
#                      metrics = ['accuracy'])

#-- plot the model
#from keras.utils import plot_model
#plot_model(my_classifier, to_file='model.png', show_shapes=True)

X_train = X_train.toarray()
# Fitting the ANN to the Training set
history = my_classifier.fit(X_train, y_train,
                            batch_size = 10, epochs = 100)

# Predicting the Test set results
X_test = X_test.toarray()
scores = my_classifier.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (my_classifier.metrics_names[1], scores[1]*100))

# Make predictions
# Predicting the Test set results
y_pred_train = my_classifier.predict(X_test)
y_pred_train = (y_pred_train > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_train)



test_df.text = train_df.text.apply(remove_symbol)
symbols = set()
for text in test_df.text.tolist():
    syms = re.findall(r'\W', text)
    for sym in syms:
        symbols.add(sym) # check for non char or non numerical values
test_df.text

test_tfidf = tfidf.fit_transform(train_df.text) 
test_tfidf = test_tfidf.toarray()


y_pred_test = my_classifier.predict(test_tfidf)

y_pred_test = (y_pred_test > 0.5)





# list all the data in history
print(history.history.keys())

import matplotlib.pyplot as plt

# Plot the accuracy for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Plot the loss for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()




my_classifier1 = Sequential()

# Adding the input layer AND the first hidden layer (Pay attention to this)
my_classifier1.add(Dense(units = 5000, kernel_initializer = 'uniform',
                        activation = 'tanh', input_dim = n_featuers))

# Adding the second hidden layer
my_classifier1.add(Dense(units = 2000, kernel_initializer = 'uniform',
                                                    activation = 'sigmoid'))

# Adding the last (output) layer
my_classifier1.add(Dense(units = 950, kernel_initializer = 'uniform',
                        activation = 'sigmoid'))

my_classifier1.add(Dense(units = 250, kernel_initializer = 'uniform',
                        activation = 'sigmoid'))

my_classifier1.add(Dense(units = 1, kernel_initializer = 'uniform',
                        activation = 'sigmoid'))

# Compiling the ANN
my_classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])


#TRYING WITH VALIDATION OPTION
history1 = my_classifier1.fit(X_train, y_train,validation_split=0.2,
                            batch_size = 10, epochs = 10)


# Predicting the Test set results
X_test1 = X_test.toarray()

scores1 = my_classifier1.evaluate(X_test1, y_test)
print("\n%s: %.2f%%" % (my_classifier1.metrics_names[1], scores1[1]*100))

# Make predictions
# Predicting the Test set results
y_pred_train1 = my_classifier1.predict(X_test1)
y_pred_train1 = (y_pred_train1 > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_train1)









y_pred_test1 = my_classifier.predict(test_tfidf)

y_pred_test1 = (y_pred_test1 > 0.5)





# list all the data in history
print(history1.history.keys())

import matplotlib.pyplot as plt

# Plot the accuracy for both train and validation set
plt.subplots() # open a new plot
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Plot the loss for both train and validation set
plt.subplots() # open a new plot
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


