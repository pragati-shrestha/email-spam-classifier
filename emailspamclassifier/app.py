# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import emoji

app = Flask(__name__)

# Load the Random Forest model and word features
with open('word_features.pkl', 'rb') as f:
    word_features = pickle.load(f)
    
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load LSTM model and tokenizer
lstm_model = load_model('spam_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

maxlength = 1000  # Set this to the same value used during training

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text for both models"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Remove emojis
    text = emoji.demojize(text)
    
    # Remove special characters and numbers
    text = re.sub(r'[@#%&*^$£!()-_+={}\[\]:;<>,.?\/\\\'"`~\d+]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def process_for_rf(text):
    """Convert text to feature vector for Random Forest"""
    cleaned_words = word_tokenize(text)
    word_count = {}
    for word in cleaned_words:
        word_count[word] = word_count.get(word, 0) + 1
    return [word_count.get(word, 0) for word in word_features]

def process_for_lstm(text):
    """Convert text to sequence for LSTM"""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlength, padding='post')
    return padded_sequences

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model_type = request.form['model']
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    if model_type == 'rf':
        # Random Forest prediction
        features = process_for_rf(processed_text)
        prediction = rf_model.predict_proba([features])[0]
        is_spam = bool(prediction[1] > 0.5)
        confidence = float(prediction[1] if is_spam else prediction[0]) * 100
    else:
        # LSTM prediction
        features = process_for_lstm(processed_text)
        prediction = lstm_model.predict(features)[0][0]
        is_spam = bool(prediction > 0.5)
        confidence = float(prediction if is_spam else 1 - prediction) * 100
    
    return jsonify({
        'prediction': 'SPAM' if is_spam else 'HAM',
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)