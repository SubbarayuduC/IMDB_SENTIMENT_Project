from flask import Flask, request, jsonify, render_template
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    review = request.form.get("review")
    if review:
        vect_text = vectorizer.transform([review])
        prediction = model.predict(vect_text)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        return render_template("index.html", prediction_text=sentiment)
    return render_template("index.html", prediction_text="Please enter a review.")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
