import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import random
import ollama  # For local LLM integration
from textblob import TextBlob  # For sentiment analysis
from googletrans import Translator  # For multi-language support
from datetime import datetime

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load intents and responses (fallback for when LLM isn’t used)
try:
    with open('intents.pkl', 'rb') as f:
        intents = pickle.load(f)
    with open('responses.pkl', 'rb') as f:
        responses = pickle.load(f)
except (FileNotFoundError, EOFError):
    intents = []
    responses = []

if intents and responses:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(intents)
else:
    vectorizer = None
    X = None

# Context storage (simple in-memory for now)
chat_context = {}

def handle_greeting(user_input):
    greetings = ['hello', 'hi', 'hey']
    return any(greeting in user_input.lower() for greeting in greetings)

def handle_goodbye(user_input):
    goodbyes = ['bye', 'goodbye', 'see you later']
    return any(goodbye in user_input.lower() for goodbye in goodbyes)

def handle_thanks(user_input):
    thanks_phrases = ['thank you', 'thanks']
    return any(phrase in user_input.lower() for phrase in thanks_phrases)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 (negative) to 1 (positive)

def translate_text(text, dest_lang='en'):
    try:
        return translator.translate(text, dest=dest_lang).text
    except Exception:
        return text  # Fallback if translation fails

def respond(user_input, user_id="default"):
    # Translate input to English if it’s not already
    try:
        detected_lang = translator.detect(user_input).lang
    except Exception:
        detected_lang = 'en'

    if detected_lang != 'en':
        user_input_en = translate_text(user_input, 'en')
    else:
        user_input_en = user_input

    # Sentiment analysis
    sentiment = get_sentiment(user_input_en)

    # Context handling
    if user_id not in chat_context:
        chat_context[user_id] = []
    context = chat_context[user_id][-3:]  # Last 3 messages for context

    # Simple rule-based responses
    if handle_greeting(user_input_en):
        response = random.choice(["Hello there!", "Hi! How can I assist you?", "Hey, nice to see you!"])
    elif handle_goodbye(user_input_en):
        response = random.choice(["Goodbye!", "See you later!", "Take care!"])
    elif handle_thanks(user_input_en):
        response = random.choice(["You're welcome!", "Happy to help!", "Anytime!"])
    else:
        # Use Ollama’s LLaMA model (or fallback to TF-IDF)
        try:
            full_prompt = f"Context: {' '.join(context)}\nUser: {user_input_en}\nRespond naturally:"
            ollama_response = ollama.generate(model='llama3', prompt=full_prompt)['response']
            response = ollama_response.strip()
        except Exception:
            # Fallback to TF-IDF if LLM fails
            if vectorizer and X is not None:
                tokens = nltk.word_tokenize(user_input_en.lower())
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                user_input_processed = ' '.join(tokens)
                user_vector = vectorizer.transform([user_input_processed])
                cosine_values = cosine_similarity(user_vector, X)
                index = np.argmax(cosine_values)
                response = responses[index]
            else:
                response = "I'm sorry, I couldn't understand that."

    # Translate back if needed
    if detected_lang != 'en':
        response = translate_text(response, detected_lang)

    # Update context
    chat_context[user_id].append(user_input_en)
    chat_context[user_id].append(response)

    return response

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    user_id = data.get('user_id', 'default')
    response = respond(user_input, user_id)
    return jsonify({'response': response, 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)