from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '',text)
    text = re.sub(r'[^a-zA-Z ]', '',text)
    text = text.lower()
    return text

@app.route('/', methods= ['GET','POST'])
def home():
    result = None
    if request.method == 'POST':
        review = request.form['review']
        cleaned_review = clean_text(review)
        vector = tfidf.transform([cleaned_review])
        prediction = model.predict(vector)[0]
        result = prediction
    return render_template('index.html', result = result)

if __name__ == '__main__':
    app.run(debug = True)