#%%
# ### Loading required modules

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%
# ### Define a callback to stop training if it reached a specific level of accuracy


class myCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
        if logs.get("acc") > 0.80:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True
            self.total_time = np.sum(self.times)
            print("Total training time was {}".format(self.total_time))


callbacks = myCallback()

#%%
# ### Loading Fashion-MNIST dataset

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (validation_images, validation_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
validation_images = validation_images.reshape(10000, 28, 28, 1)
print(train_images.shape, validation_images.shape)

#%%
# ### Visualizing the examples

plt.imshow(train_images[200, :, :, 0], cmap="gray")
print(train_labels[200])
print(train_images[200, :, :, 0])

#%%
# ### Feature normalization

train_images = train_images / 255.0
validation_images = validation_images / 255.0

#%%
# ### Defining the neural network's architecture

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        # tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # tf.keras.layers.MaxPool2D(2, 2),
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

model.fit(train_images, train_labels, epochs=2, callbacks=[callbacks])

#%%
# ### Model evaluation

validation_loss = model.evaluate(validation_images, validation_labels)

#%%
# ### Assessing the predictions by the trained model

predictions = model.predict(validation_images)
predictions[100]
print(validation_labels[100])
plt.imshow(validation_images[100, :, :, 0], cmap="gray")
# print(validation_labels[:100])

#%%
# ### Visualizing convolutions step by step

fig, ax = plt.subplots(3, 4)
FIRST_IMAGE = 2
SECOND_IMAGE = 8
THIRD_IMAGE = 30
CONVOLUTION_NUMBER = 7
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
for x in range(4):
    f1 = activation_model.predict(validation_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    ax[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap="gray")
    ax[0, x].grid = False
    f2 = activation_model.predict(validation_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    ax[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap="gray")
    ax[1, x].grid = False
    f3 = activation_model.predict(validation_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    ax[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap="gray")
    ax[2, x].grid = False
