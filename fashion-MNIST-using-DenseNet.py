#%%
# ### Loading required modules

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#%%
# ### Define a callback to stop training if it reached a specific level of accuracy


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("acc") > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

#%%
# ### Loading Fashion-MNIST dataset

fashion_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (validation_images, validation_labels) = fashion_mnist.load_data()

#%%
# ### Visualizing the examples

plt.imshow(train_images[200], cmap="gray")
print(train_labels[15])
print(train_images[15])

#%%
# ### Feature normalization

train_images = train_images / 255.0
validation_images = validation_images / 255.0

#%%
# ### Defining the neural network's architecture

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

#%%
# ### Training settings

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

#%%
# ### Training the model on train data

model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])

#%%
# ### Model evaluation

model.evaluate(validation_images, validation_labels)

#%%
# ### Assessing the predictions by the trained model

predictions = model.predict(validation_images)
predictions[200]
print(validation_labels[240])
plt.imshow(validation_images[240], cmap="gray")
