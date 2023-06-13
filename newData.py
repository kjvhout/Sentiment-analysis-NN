from keras.utils.data_utils import pad_sequences
import pickle
from keras.models import load_model

# Load the models and tokenizers
lstm_model = load_model("sentiment_model_RecNN.h5")
lstm_tokenizer = pickle.load(open("sentiment_tokenizer_RecNN.pkl", "rb"))
cnn_model = load_model("sentiment_model_CNN.h5")
cnn_tokenizer = pickle.load(open("sentiment_tokenizer_CNN.pkl", "rb"))

while True:
    # Get user input
    text = input("Enter text for sentiment analysis (or 'quit' to exit): ")

    if text.lower() == "quit":
        break

    # Tokenize and pad the input text
    lstm_sequences = lstm_tokenizer.texts_to_sequences([text])
    lstm_padded_sequences = pad_sequences(lstm_sequences, padding='post', maxlen=50)
    cnn_sequences = cnn_tokenizer.texts_to_sequences([text])
    cnn_padded_sequences = pad_sequences(cnn_sequences, maxlen=50)

    # Perform sentiment analysis with LSTM model
    lstm_prediction = lstm_model.predict(lstm_padded_sequences)[0][0]
    lstm_sentiment = "Positive" if lstm_prediction >= 0.5 else "Negative"

    # Perform sentiment analysis with CNN model
    cnn_prediction = cnn_model.predict(cnn_padded_sequences)[0][0]
    print(cnn_prediction)
    cnn_sentiment = "Positive" if cnn_prediction >= 0.5 else "Negative"

    # Print the sentiment and confidence for each model
    print("RecNN Model:")
    print("Sentiment: ", lstm_sentiment)

    print("CNN Model:")
    print(cnn_prediction)
    print("Sentiment: ", cnn_sentiment)
