#%% [markdown]
# ## Horse vs. Human Classification using Transfer Learning and Dropout Regularization

#%% [markdown]
# ### Loading required libraries
import os
import tensorflow as tf
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

output_notebook()

#%% [markdown]
# ### Load the pretrained Inception model and lock the layers
local_weights_file = "./inception_v3.h5"
pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None
)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()

#%% [markdown]
# ### Get the last non-FC layer output
last_layer = pre_trained_model.get_layer("mixed7")
print("Last layer output shape:", last_layer.output_shape)
last_output = last_layer.output


#%% [markdown]
# ### Define a callback to stop training when the accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("acc") > 0.999:
            print("\nReached 99.9% accuracy. So, cancelling the training!")
            self.model.stop_training = True


#%% [markdown]
# ### Define the structure of addon DNN
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(pre_trained_model.input, x)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
    loss="binary_crossentropy",
    metrics=["acc"],
)
model.summary()

#%% [markdown]
# ### Defining training and validation images directories as path variables
train_horse_dir = os.path.join("./Training Data/Horse or Human/horses")
train_human_dir = os.path.join("./Training Data/Horse or Human/humans")
validation_horse_dir = os.path.join("./Validation Data/Horse or Human/horses")
validation_human_dir = os.path.join("./Validation Data/Horse or Human/humans")

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
# ### Generating training and validation data
training_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
validation_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)
training_data = training_data_gen.flow_from_directory(
    "./Training Data/Horse or Human",
    batch_size=20,
    class_mode="binary",
    target_size=(150, 150),
)
validation_data = validation_data_gen.flow_from_directory(
    "./Validation Data/Horse or Human",
    batch_size=20,
    class_mode="binary",
    target_size=(150, 150),
)


#%% [markdown]
# ### Training the model
callbacks = myCallback()
history = model.fit_generator(
    training_data,
    validation_data=validation_data,
    steps_per_epoch=8,
    epochs=20,
    validation_steps=8,
    verbose=2,
    callbacks=[callbacks],
)

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
