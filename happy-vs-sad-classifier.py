#%% [markdown]
# ## Happy vs. Sad classifier using convolutional neural network

#%% [markdown]
# ### Loading required libraries
import tensorflow as tf
import os
import zipfile
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as image

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% [markdown]
# ### Setting constants
DESIRED_ACCURACY = 0.95

#%% [markdown]
# ### Extracting training data from zip archive
zip_ref = zipfile.ZipFile("./happy-or-sad.zip", "r")
zip_ref.extractall("./happy-or-sad")
zip_ref.close()

#%% [markdown]
# ### Taking a look at the image files and their size
train_happy_dir = os.path.join("./Training Data/Happy or Sad/happy")
train_sad_dir = os.path.join("./Training Data/Happy or Sad/sad")
train_happy_names = os.listdir("./Training Data/Happy or Sad/happy")
train_sad_names = os.listdir("./Training Data/Happy or Sad/sad")
print("Total training happy images: {}".format(len(train_happy_names)))
print("Total training sad images: {}".format(len(train_sad_names)))
happy_sample = random.choice(train_happy_names)
happy_pic = os.path.join(train_happy_dir, happy_sample)
sad_sample = random.choice(train_sad_names)
sad_pic = os.path.join(train_sad_dir, sad_sample)
happy_img = image.imread(happy_pic)
print(happy_img.shape)
sad_img = image.imread(sad_pic)
plt.imshow(happy_img)
plt.show()


#%% [markdown]
# ### Defining a callback class to stop training when reached to desired accuracy
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, desired_accuracy):
        self.desired_accuracy = desired_accuracy

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_start_time)
        if logs.get("acc") > self.desired_accuracy:
            print(
                "\nReached {} accuracy. So, cancelling training!".format(
                    self.desired_accuracy
                )
            )
            self.model.stop_training = True
            self.total_time = np.sum(self.times)
            print("Total training time was {}".format(self.total_time))


callbacks = myCallback(DESIRED_ACCURACY)

#%% [markdown]
# ### Structuring the ConvNet
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

#%%
# ### Let's take a look at model structure
model.summary()

#%% [markdown]
# ### Model compile settings
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss="binary_crossentropy",
    metrics=["acc"],
)

#%% [markdown]
# ### Generating labeled data for training using images in directories
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    "./Training Data/Happy or Sad", target_size=(150, 150), batch_size=10, class_mode="binary"
)

#%% [markdown]
# ### Training the model
model.fit_generator(
    train_generator, steps_per_epoch=None, epochs=10, verbose=1, callbacks=[callbacks]
)

#%% [markdown]
# ### Let's test the model with an external image
path = "./Validation Data/External Images/happy2.png"
validation_image = tf.keras.preprocessing.image.load_img(
    path, target_size=(150, 150)
)  # this is a PIL image
x = tf.keras.preprocessing.image.img_to_array(
    validation_image
)  # Numpy array with shape (150, 150, 3)
x = np.expand_dims(x, axis=0)  # Numpy array with shape (1, 150, 150, 3)
x /= 255  # Rescale by 1/255
plt.imshow(validation_image)
plt.show()
prediction = model.predict(x)
if prediction > 0.5:
    print("Happy!")
else:
    print("Sad!")
