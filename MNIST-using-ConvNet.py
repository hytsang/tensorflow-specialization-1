#%% [markdown]
# ### Loading required modules

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

#%% [markdown]
# ### Define a callback to stop training if we reached 99.8% accuracy


class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, desired_accuracy):
        self.desired_accuracy = desired_accuracy

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
        if logs.get("acc") > self.desired_accuracy / 100:
            print(
                "\nReached {}% accuracy so cancelling training!".format(
                    self.desired_accuracy
                )
            )
            self.model.stop_training = True
            self.total_time = np.sum(self.times)
            print("Total training time was {}".format(self.total_time))


callbacks = myCallback(99.8)

#%% [markdown]
# ### Loading MNIST dataset

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (validation_images, validation_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
validation_images = validation_images.reshape(10000, 28, 28, 1)

#%% [markdown]
# ### Visualizing the examples

example_index = 4000
plt.imshow(train_images[example_index, :, :, 0], cmap="gray")
print(train_labels[example_index])

#%% [markdown]
# ### Feature normalization

train_images = train_images / 255.0
validation_images = validation_images / 255.0

#%%
# ### Defining the neural network's architecture

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

#%%
# ### Training settings
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

#%%
# ### Training the model on train data

model.fit(x=train_images, y=train_labels, epochs=10, callbacks=[callbacks])
