#%% [markdown]
# ## Horse vs. Human Binary Classifier with ConvNet

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
local_zip = "./horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("./horse-or-human")
zip_ref.close()

#%% [markdown]
# ### Extract the zip file containing validation data
local_zip = "./validation-horse-or-human.zip"
zip_ref = zipfile.ZipFile(local_zip, "r")
zip_ref.extractall("./validation-horse-or-human")
zip_ref.close()

#%% [markdown]
# ### Defining training and validation images directories as path variables
train_horse_dir = os.path.join("./horse-or-human/horses")
train_human_dir = os.path.join("./horse-or-human/humans")
validation_horse_dir = os.path.join("./validation-horse-or-human/horses")
validation_human_dir = os.path.join("./validation-horse-or-human/humans")

#%% [markdown]
# ### Let's see how the filenames look like
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
print(train_horse_names[:10])
print(train_human_names[:10])
validation_horse_names = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)
print(validation_horse_names[:10])
print(validation_human_names[:10])

#%% [markdown]
# ### Total number of training and validation images
print("Total training horse images: ", len(train_horse_names))
print("Total training human images: ", len(train_human_names))
print("Total validation horse images: ", len(validation_horse_names))
print("Total validation human images: ", len(validation_human_names))

#%% [markdown]
# ### Let's look at some of the pictures
nrows, ncols = 4, 4
pic_index = 0
random.seed(100)
horse_sample = random.sample(train_horse_names, 8)
horse_pics = [os.path.join(train_horse_dir, name) for name in horse_sample]
human_sample = random.sample(train_human_names, 8)
human_pics = [os.path.join(train_human_dir, name) for name in human_sample]
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for index, image_path in enumerate(horse_pics + human_pics):
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
            16, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

#%% [markdown]
# ### Get the model summary
model.summary()

#%% [markdown]
# ### Compiling the model
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss="binary_crossentropy",
    metrics=["acc"],
)

#%% [markdown]
# ### Generating training and validation labeled data from images in directories
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    "./horse-or-human", target_size=(300, 300), batch_size=128, class_mode="binary"
)
validation_generator = validation_datagen.flow_from_directory(
    "./validation-horse-or-human",
    target_size=(300, 300),
    batch_size=32,
    class_mode="binary",
)

#%% [markdown]
# ### Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8,
)

#%% [markdown]
# ### Test classifier with real images
path = "./Validation Data/External Images/horse_test.jpg"
img = tf.keras.preprocessing.image.load_img(path, target_size=(300, 300))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
prediction = model.predict(x)
if prediction > 0.5:
    print("It is a human.")
else:
    print("It is a horse.")

#%% [markdown]
# ### Show convolutional layers representations
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(
    inputs=model.input, outputs=successive_outputs
)
img_path = random.choice(horse_pics + human_pics)
img = tf.keras.preprocessing.image.load_img(
    img_path, target_size=(300, 300)
)  # this is a PIL image
x = tf.keras.preprocessing.image.img_to_array(
    img
)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
x /= 255  # Rescale by 1/255

successive_feature_maps = visualization_model.predict(
    x
)  # Inject image into model to get the intermediary representations
layer_names = [layer.name for layer in model.layers[1:]]  # Names of the layers to use

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:  # Do this for non-FC layers
        n_features = feature_map.shape[-1]  # number of features in feature map
        size = feature_map.shape[
            1
        ]  # The feature map has shape (1, size, size, n_features)
        display_grid = np.zeros(
            (size, size * n_features)
        )  # We will tile our images in this matrix
        for i in range(n_features):
            # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype("uint8")
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x
        # Display the grid
        scale = 20.0 / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="viridis")

#%% [markdown]
# ### Cleanup memory
os.kill(os.getpid(), signal.SIGKILL)
