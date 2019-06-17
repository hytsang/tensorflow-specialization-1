#%% [markdown]
# ### Loading required libraries
import tensorflow as tf
import numpy as np

#%% [markdown]
# ### Defining and compiling the single layer and single neuron neural network (~ linear regression)
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")

#%% [markdown]
# ### Training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#%% [markdown]
# ### Train the model
model.fit(xs, ys, epochs=50)

#%% [markdown]
# ### Test the model output
print(model.predict([10.0]))
