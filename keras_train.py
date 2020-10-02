# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# Author:   Carlos Miguel Sayao
# Purpose:  Keras implementation of MNIST
# Sources:  https://www.tensorflow.org/tutorials/quickstart/beginner
#           http://yann.lecun.com/exdb/mnist/
#           https://stackoverflow.com/a/47227886
# Date:     Jun 4, 2020
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


import tensorflow as tf
# Disable GPU warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist = tf.keras.datasets.mnist

# These are 3D arrays vs the 2D arrays from class
# x_train.shape: (60000, 28, 28), x_test.shape: (10000, 28, 28)
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # Load
x_train, x_test = x_train / 255.0, x_test / 255.0           # Normalize

# Build tf.keras.Sequential model by stacking layers. Choose optimizer and
# lsos function for training:
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# See sample
print(f"y_test[:1]\n{y_test[:1]}")

# Return a vector of logits scores, one for each class
predictions = model(x_train[:1]).numpy()
print(f"\npredictions\n{predictions}")

# The tf.nn.softmax function converts logits to probabilities for each class:
probabilities = tf.nn.softmax(predictions).numpy()
print(f"\nprobabilities\n{probabilities}")

# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a
# True index and returns a scaler loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# This loss is equal to the negative log probability of the true class:
# It is zero if the model is sure of the correct class.
print(f"\n{loss_fn(y_train[:1], predictions).numpy()}")

# Configure the model for training.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# The Model.fit method adjusts model parameters to minimize loss:
model.fit(x_train, y_train, epochs=5)

# The Model.evaluate checks model performance, usually on validation or test set.
model.evaluate(x_test, y_test, verbose=2)

# If you want your model to return a probability, you can wrap the trained
# model, and attach the softmax to it:
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

tf.keras.utils.plot_model(model, "my_first_model.png")
tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
model.summary()


print()
