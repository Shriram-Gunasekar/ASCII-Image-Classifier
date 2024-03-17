# Function to preprocess data
def preprocess_data(image_to_ascii_data):
    X = [data[0] for data in image_to_ascii_data]
    y = [data[1] for data in image_to_ascii_data]
    return X, y

