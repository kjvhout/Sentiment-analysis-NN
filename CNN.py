from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Dense, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pymongo import MongoClient
from time import time
from matplotlib import pyplot as plt
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

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

# save the tokenizer for later use
pickle.dump(tokenizer, open("sentiment_tokenizer_CNN.pkl", "wb"))

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to a fixed length
max_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=50)


# Convert labels to numpy array
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build the CNN model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=32, input_length=50))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Fit the model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=1)

end_time = time()

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

seconds_elapsed = end_time - start_time
print('Amount of time needed to complete: ' + str(seconds_elapsed))

# Save the model for later use
model.save("sentiment_model_CNN_test.h5")
# save the tokenizer for later use
pickle.dump(tokenizer, open("sentiment_tokenizer_CNN_test.pkl", "wb"))
