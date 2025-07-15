# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# import pickle
# import requests
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = Flask(__name__)

# # Load model and tokenizer
# model = tf.keras.models.load_model("sentiment_model.h5")
# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=["POST"])
# def predict():
#     text = request.form["text"]
#     sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(sequence, padding='post', maxlen=200)
#     prediction = model.predict(padded_sequence)
#     sentiment = 'positive' if prediction > 0.5 else 'negative'
#     return sentiment

#     # Optional Emoji API Example (fun interactivity)
#     emoji_api = f"https://emojihub.yurace.pro/api/random"  # public emoji API
#     emoji_response = requests.get(emoji_api).json()
#     emoji_char = emoji_response.get("emoji", "🎭")

#     return render_template("index.html", prediction=sentiment, emoji=emoji_char, text=text, confidence=confidence)

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Predict sentiment function
def predict_sentiment(text):
    # Optional enhancement for short text
    if len(text.split()) < 3:
        text = "This is " + text

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=200)

    # Predict
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = 'Positive 😊' if prediction >= 0.5 else 'Negative 😞'
    confidence = round(prediction if prediction >= 0.5 else 1 - prediction, 2)
    return sentiment, confidence

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    sentiment, confidence = predict_sentiment(user_input)
    return render_template("index.html", prediction=sentiment,
                           confidence=confidence, user_input=user_input)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
 