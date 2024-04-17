from flask import Flask, render_template, request
import random
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load intent data from JSON file
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

# Synthetic dataset
dataset = []
for intent in intents_data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['responses']
    for pattern in patterns:
        dataset.append((pattern, tag))
    for response in responses:
        dataset.append((response, tag))

# Shuffle the dataset
random.shuffle(dataset)

# Preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

# Model Building
X, y = zip(*dataset)
vectorizer = CountVectorizer(analyzer=preprocess)
classifier = MultinomialNB()
model = make_pipeline(vectorizer, classifier)
model.fit(X, y)

# Function to fetch response based on intent
def get_response(message):
    intent = model.predict([message])[0]
    for data in intents_data['intents']:
        if data['tag'] == intent:
            return random.choice(data['responses'])
    return "I'm sorry, I don't understand."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']
    bot_response = get_response(user_message)
    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
