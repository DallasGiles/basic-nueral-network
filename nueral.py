import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Specify the local path to the MNIST dataset
local_path = "mnist.npz"  # Ensure this file is in the same directory as your script

# Load the MNIST dataset from the local file
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=local_path)

# Normalize the data (scale pixel values to the range [0, 1])
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D vectors
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax'),  # Output layer with 10 neurons (one for each digit)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Predict a digit
sample_image = X_test[0].reshape(1, 28, 28)
prediction = np.argmax(model.predict(sample_image))
print(f"Predicted digit: {prediction}")