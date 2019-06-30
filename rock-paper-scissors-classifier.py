#%% [markdown]
# ## Rock, Paper, Scissors Categorical Classifier with ConvNet

#%% [markdown]
# ### Loading required libraries
import os, signal
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
import random

#%% [markdown]
# ### Extract the zip file containing train data
local_zip = "./training-rps.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("./Training Data/Rock, Paper, Scissors")
zip_ref.close()

#%% [markdown]
# ### Extract the zip file containing validation data
local_zip = "./validation-rps.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("./Validation Data/Rock, Paper, Scissors")
zip_ref.close()

#%% [markdown]
# ### Defining training and validation images directories as path variables
train_rock_dir = os.path.join("./Training Data/Rock, Paper, Scissors/rock")
train_paper_dir = os.path.join("./Training Data/Rock, Paper, Scissors/paper")
train_scissors_dir = os.path.join("./Training Data/Rock, Paper, Scissors/scissors")
validation_rock_dir = os.path.join("./Validation Data/Rock, Paper, Scissors/rock")
validation_paper_dir = os.path.join("./Validation Data/Rock, Paper, Scissors/paper")
validation_scissors_dir = os.path.join(
    "./Validation Data/Rock, Paper, Scissors/scissors"
)

#%% [markdown]
# ### Let's see how the filenames look like
train_rock_names = os.listdir(train_rock_dir)
train_paper_names = os.listdir(train_paper_dir)
train_scissors_names = os.listdir(train_scissors_dir)
print(train_rock_names[:10])
print(train_paper_names[:10])
print(train_scissors_names[:10])
validation_rock_names = os.listdir(validation_rock_dir)
validation_paper_names = os.listdir(validation_paper_dir)
validation_scissors_names = os.listdir(validation_scissors_dir)
print(validation_rock_names[:10])
print(validation_paper_names[:10])
print(validation_scissors_names[:10])

#%% [markdown]
# ### Total number of training and validation images
print("Total training rock images: ", len(train_rock_names))
print("Total training paper images: ", len(train_paper_names))
print("Total training scissors images: ", len(train_scissors_names))
print("Total validation rock images: ", len(validation_rock_names))
print("Total validation paper images: ", len(validation_paper_names))
print("Total validation scissors images: ", len(validation_scissors_names))

#%% [markdown]
# ### Let's look at some of the pictures
nrows, ncols = 6, 4
pic_index = 0
random.seed(100)
rock_sample = random.sample(train_rock_names, 8)
rock_pics = [os.path.join(train_rock_dir, name) for name in rock_sample]
paper_sample = random.sample(train_paper_names, 8)
paper_pics = [os.path.join(train_paper_dir, name) for name in paper_sample]
scissors_sample = random.sample(train_scissors_names, 8)
scissors_pics = [os.path.join(train_scissors_dir, name) for name in scissors_sample]
fig = plt.gcf()
fig.set_size_inches(ncols * 2, nrows * 2)
for index, image_path in enumerate(rock_pics + paper_pics + scissors_pics):
    sp = plt.subplot(nrows, ncols, index + 1)
    sp.axis("Off")
    img = mpimage.imread(image_path)
    plt.imshow(img)
plt.show()

#%% [markdown]
# ### Structuring the ConvNet architecture
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

#%% [markdown]
# ### Get the model summary
model.summary()

#%% [markdown]
# ### Compiling the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

#%% [markdown]
# ### Generating training and validation labeled data from images in directories
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    "./Training Data/Rock, Paper, Scissors",
    target_size=(300, 300),
    class_mode="categorical",
)
validation_generator = validation_datagen.flow_from_directory(
    "./Validation Data/Rock, Paper, Scissors",
    target_size=(300, 300),
    class_mode="categorical",
)

#%% [markdown]
# ### Train the model
history = model.fit_generator(
    train_generator, epochs=25, verbose=1, validation_data=validation_generator
)

#%% [markdown]
# ### Plot training and validation accuracy
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))
plt.plot(epochs, acc, "r", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend(loc=0)
plt.show()

#%% [markdown]
# ### Test classifier with real images
path = "./Validation Data/External Images/Rock, Paper, Scissors/scissors_test.jpg"
img = tf.keras.preprocessing.image.load_img(path, target_size=(300, 300))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
prediction = model.predict(x)
print(prediction)

#%% [markdown]
# ### Save the Model in a File and Cleanup memory
model.save("rps.h5")
os.kill(os.getpid(), signal.SIGKILL)
