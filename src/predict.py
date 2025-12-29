import pickle

# Load model and vectorizer
with open("../models/phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Sample email
email = input("Enter email text: ")

email_vectorized = vectorizer.transform([email])
prediction = model.predict(email_vectorized)

if prediction[0] == 1:
    print("Phishing Email Detected")
else:
    print("Legitimate Email")

