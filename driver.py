# Example usage
def main():
    # Dummy data for illustration (replace with your own dataset)
    # Assume image_to_ascii_data is a list of (ASCII text, label) tuples
    image_to_ascii_data = [(image_to_ascii('example_image.jpg'), 'cat'),
                           (image_to_ascii('example_image2.jpg'), 'dog'),
                           (image_to_ascii('example_image3.jpg'), 'cat')]

    # Preprocess data
    X, y = preprocess_data(image_to_ascii_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract features using TF-IDF vectorization
    X_train_vectorized, X_test_vectorized = extract_features(X_train, X_test)

    # Train the model
    clf = train_model(X_train_vectorized, y_train)

    # Evaluate the model
    accuracy = evaluate_model(clf, X_test_vectorized, y_test)
    print(f'Accuracy: {accuracy}')

    # Make predictions
    new_ascii_text = image_to_ascii('new_image.jpg')
    prediction = predict(clf, new_ascii_text)
    print(f'Prediction: {prediction}')

if __name__ == "__main__":
    main()
