# Function to make predictions
def predict(clf, new_ascii_text):
    new_ascii_vectorized = vectorizer.transform([new_ascii_text])
    prediction = clf.predict(new_ascii_vectorized)
    return prediction[0]

