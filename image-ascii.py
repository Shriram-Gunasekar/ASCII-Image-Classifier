from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to convert image to ASCII
def image_to_ascii(image_path, width=100):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio / 2.5)
    img = img.resize((width, height))
    pixels = np.array(img)
    chars = '@%#*+=-:. '  # ASCII characters from dark to light
    ascii_str = ''
    for row in pixels:
        for pixel in row:
            ascii_str += chars[pixel // 32]
        ascii_str += '\n'
    return ascii_str

