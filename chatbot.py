from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQs
df = pd.read_csv('faq.csv')

# Preprocess questions
questions = df['Question'].values
answers = df['Answer'].values

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_answer(user_input):
    user_input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vec, X)
    best_match = similarities.argmax()
    if similarities[0][best_match] < 0.2:
        return "Sorry, I couldn't find a relevant answer. Please contact the college office."
    return answers[best_match]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_msg = request.form['msg']
    response = get_answer(user_msg)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
