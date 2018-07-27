import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print("tensorflow version:", tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# For reference later when plotting
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data pre-processing
""" To look at feature scales...
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

# Feature scaling as largest scaling ranges from 0 (black) - 255 (white)
train_images = train_images / 255.
test_images = test_images / 255.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # Percentage of images correctly classified
)

# Training the model
model.fit(train_images, train_labels, epochs=5)  # Starts training

# Accuracy evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy: {}".format(test_accuracy))

"""
    Test accuracy was lower than training accuracy suggesting possible over-fitting.
        - Decrease test size of train in (train:test) ratio to fix this. 
"""

# Making predictions
predictions = model.predict(test_images)

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} ({})".format(
        class_names[predicted_label],
        class_names[true_label]),
        color=color
    )
plt.show()
