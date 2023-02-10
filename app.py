import pickle
import string
import sklearn
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/')
def index():
    words = os.path.join(app.config['UPLOAD_FOLDER'], 'spamwords.png')
    return render_template("index.html", user_image=words)


@app.route('/classifier')
def recommend_ui():
    return render_template("classifier.html")


@app.route('/word_classifier', methods=['post'])
def recommend():
    email = ""
    msg = []
    input_msg = request.form.get('user_sms')
    msg.append(input_msg)
    for word in msg:
        email += word

    print(msg)
    print(email)

    result = []
    input_sms = request.form.get('user_sms')
    print(input_sms)
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    predict = model.predict(vector_input)[0]
    # 4. Display

    if predict == 1:
        print("Spam")
        result.append('Spam')
    else:
        print("Not-Spam")
        result.append('Not-Spam')

    print(result)

    return render_template('classifier.html', email=email, data=result)


@app.route('/msg', methods=['post'])
def massage():
    msg = []
    input_msg = request.form.get('user_sms')
    msg.append(input_msg)
    print(msg)

    return render_template('classifier.html', data=msg)


if __name__ == '__main__':
    app.run(debug=True)
