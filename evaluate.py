# Function to evaluate the model
def evaluate_model(clf, X_test_vectorized, y_test):
    y_pred = clf.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

