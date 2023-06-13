from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import pandas as pd
from pymongo import MongoClient
from keras.utils.data_utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from time import time
import pickle

# Set the limit for the number of reviews to fetch from MongoDB
LIMIT = 500000

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client.reviews
collection = db["reviews"]

# Fetch reviews from the database and store them in a DataFrame
reviews = collection.find(limit=LIMIT)
start_time = time()
data = pd.DataFrame(list(reviews))

# Extract the text reviews and corresponding labels from the DataFrame
texts = data['review']
labels = data['positive']


X_train, X_test , y_train, y_test = train_test_split(texts, labels, train_size=0.5, random_state=1)

from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
max_words=5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
max_len = 50

#Model creation

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['acc'])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000, binary=False)
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# Fit the model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

#Evaluate the model
score = model.evaluate(X_test, y_test, verbose=1)

end_time = time()

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

from matplotlib import pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

seconds_elapsed = end_time - start_time
print('Amount of time needed to complete: ' + str(seconds_elapsed))

liveTest = ['It was a really bad expierence, never comming back', 'We had a nice stay at this Hotel, Defenetly comming back','We had a nice experience']
liveTest = tokenizer.texts_to_sequences(liveTest)
liveTest = pad_sequences(liveTest, padding='post', maxlen=max_len)
print(model.predict(liveTest, verbose=0))

# Save the model for later use
model.save("sentiment_model_CNN_test.h5")
# save the tokenizer for later use
pickle.dump(tokenizer, open("sentiment_tokenizer_CNN_test.pkl", "wb"))