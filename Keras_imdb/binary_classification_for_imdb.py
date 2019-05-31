#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:02:18 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

# import matplotlib.pyplot as plt
from keras.datasets import imdb


# In[3]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# In[4]:


word_index = imdb.get_word_index()
reverse_word_index = dict([(key, value) for (key, value) in word_index.items()])


# In[5]:


decoded_review = " ".join([reverse_word_index.get(i-3, "?") for i in train_data[0]])


# In[6]:


decoded_review


# In[7]:


reverse_word_index


# In[8]:


import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros(((len(sequences)), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)


# In[9]:


X_train[0]


# In[10]:


train_data[0]


# In[11]:


y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


# In[12]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))


# In[13]:


# compiling the model
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])


# In[14]:


# set aside a validation set
x_val = X_train[:10000]
partial_x_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[15]:


history = model.fit(partial_x_train, 
                    partial_y_train, 
                    epochs=20, 
                    batch_size=512, 
                    validation_data=(x_val, y_val))


# In[16]:


history_dict = history.history
history_dict.keys()


# In[ ]:


#loss_values = history_dict["loss"]
#val_loss_values = history_dict["val_loss"]
#epochs = range(1, 21)
#
#plt.plot(epochs, loss_values, "bo", label="Training Loss")
#plt.plot(epochs, val_loss_values, "b", label="Validation Loss")
#plt.title("Train and Validation Loss")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()


# In[ ]:




