# Function to train the model
def train_model(X_train_vectorized, y_train):
    clf = SVC(kernel='linear')
    clf.fit(X_train_vectorized, y_train)
    return clf

