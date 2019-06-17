#%% [markdown]
# ### Loading required libraries
import os
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

output_notebook()

#%% [markdown]
# ### Set the training and validation directories
training_dir = "./Training Data/Cats vs Dogs"
validation_dir = "./Validation Data/Cats vs Dogs"
running_dir = "./Validation Data/External Images/Cats vs Dogs"

#%% [markdown]
# ### Take a look at the filenames
train_cat_file_names = os.listdir(os.path.join(training_dir, "cats"))
train_dog_file_names = os.listdir(os.path.join(training_dir, "dogs"))
print(random.sample(train_cat_file_names, 10))
print(random.sample(train_dog_file_names, 10))

#%% [markdown]
# ### Total number of cat and dog images in train and validation datasets
print(
    "We have {0} cat images and {1} dog images in the training set.".format(
        len(train_cat_file_names), len(train_dog_file_names)
    )
)

#%% [markdown]
# ### Take a look at some of the training images
rand_cat_file_names = random.sample(train_cat_file_names, 8)
rand_dog_file_names = random.sample(train_dog_file_names, 8)
rand_cat_pix = [
    os.path.join(training_dir, "cats", file_name) for file_name in rand_cat_file_names
]
rand_dog_pix = [
    os.path.join(training_dir, "dogs", file_name) for file_name in rand_dog_file_names
]
nrows = 4
ncols = 4
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(rand_cat_pix + rand_dog_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis("off")
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

#%% [markdown]
# ### The CNN model structure
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(150, 150, 3)
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
# ### Check model summary
model.summary()

#%% [markdown]
# ### Model settings and compilation
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss="binary_crossentropy",
    metrics=["acc"],
)

#%% [markdown]
# ### Generating training and validation data
training_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
validation_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255.
)
training_data = training_data_gen.flow_from_directory(
    training_dir, batch_size=128, class_mode="binary", target_size=(150, 150)
)
validation_data = validation_data_gen.flow_from_directory(
    validation_dir, batch_size=32, class_mode="binary", target_size=(150, 150)
)


#%% [markdown]
# ### Training the model
history = model.fit_generator(
    training_data,
    validation_data=validation_data,
    steps_per_epoch=8,
    epochs=50,
    validation_steps=8,
    verbose=1,
)


#%% [markdown]
# ### Running the model
running_fnames = os.listdir(running_dir)
running_pix = [os.path.join(running_dir, fname) for fname in running_fnames]
for picture in running_pix:
    sample_image = tf.keras.preprocessing.image.load_img(
        picture, target_size=(150, 150)
    )
    x = tf.keras.preprocessing.image.img_to_array(sample_image)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    plt.imshow(sample_image)
    plt.axis("off")
    plt.show()
    prediction = model.predict(x)
    if prediction > 0.5:
        print("A dog!")
    else:
        print("A cat!")

#%% [markdown]
# ### Visualizing intermediary representations
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(
    inputs=model.input, outputs=successive_outputs
)
rand_cat_fname = random.choice(train_cat_file_names)
sample_image = tf.keras.preprocessing.image.load_img(
    os.path.join(training_dir, "cats", rand_cat_fname), target_size=(150, 150)
)
x = tf.keras.preprocessing.image.img_to_array(sample_image)
x = np.expand_dims(x, axis=0)
x /= 255.
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers[1:]]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))

        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype("uint8")
            display_grid[:, i * size : (i + 1) * size] = x

        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="gray")

#%% [markdown]
# ### Evaluating accuracy and loss of the model
training_accuracy = history.history["acc"]
training_loss = history.history["loss"]
validation_accuracy = history.history["val_acc"]
validation_loss = history.history["val_loss"]
epochs = range(len(training_accuracy))
p = figure(
    plot_width=720,
    plot_height=360,
    title="Training vs Validation Accuracy",
    toolbar_location="above",
)
p.line(epochs, training_accuracy, line_width=2, color="red", legend="Training")
p.line(epochs, validation_accuracy, line_width=2, color="blue", legend="Validation")
p.xaxis.axis_label = "# of Epochs"
p.yaxis.axis_label = "Accuracy (%)"
show(p)
p = figure(
    plot_width=720,
    plot_height=360,
    title="Training vs Validation Loss",
    toolbar_location="above",
)
p.line(epochs, training_loss, line_width=2, color="red", legend="Training")
p.line(epochs, validation_loss, line_width=2, color="blue", legend="Validation")
p.xaxis.axis_label = "# of Epochs"
p.yaxis.axis_label = "Loss"
show(p)


#%%
