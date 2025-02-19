from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


app = Flask(__name__)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def process_data(text):
    text = text.lower()
    text = re.sub('<br />', ' ', text)
    text = re.sub(r"https\S+|http\S+|www\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    cleaned_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(cleaned_text)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        review = request.form.get('review_text')
        if not review:
            return render_template('index.html', error="Please enter a movie review.")
        
        cleaned_review = process_data(review)
        review_tfidf = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_tfidf)[0]
        sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"
        return render_template('index.html', review=review, cleaned_review=cleaned_review, sentiment=sentiment)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
