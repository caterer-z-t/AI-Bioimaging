#!/usr/bin/env python
# coding: utf-8

# In[8]:
#
# Multi Layer Perceptron for Binary Classification
import tensorflow as tf
from matplotlib import pyplot
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import time
import os
import tempfile


# In[9]:


# Load the Datasets
path = "raw_mosaic_pixeldata_NN.csv"
df = read_csv(path, header=None)

feature_path = "ETC_RFE_ANOVA.csv"
feature_df = read_csv(feature_path, header=None)
feature_array_triple = feature_df.to_numpy()
print(feature_array_triple)

feature_path = "12_features.csv"
feature_df = read_csv(feature_path, header=None)
feature_array_twelve = feature_df.to_numpy()
print(feature_array_twelve)

feature_path = "LIT_Bands_Intersection_Feature_Selection.csv"
feature_df = read_csv(feature_path, header=None)
feature_array_LIT_intersect = feature_df.to_numpy()
print(feature_array_LIT_intersect)

feature_path = "Literature_Bands_61.csv"
feature_df = read_csv(feature_path, header=None)
feature_array_LIT_61 = feature_df.to_numpy()
print(feature_array_LIT_61.size)


# Function used to return just the values in the list, can be used for 104 and 12 feature lists
def array_values(list):
    for i in list:
        return i


# In[10]:


# Split into input/output columns
# Encode Strings to integers
# Split into Training and Test Datasets
X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype("float32")
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=17
)
n_features = X_train.shape[1]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:

# Create the model first time

# Define Model
# 767 Bands
N = 100
model_1 = Sequential()
model_1.add(
    Dense(
        N,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        input_shape=(n_features,),
    )
)
model_1.add(
    Dense(
        2 * N,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        kernel_initializer="he_normal",
    )
)
model_1.add(
    Dense(
        3 * N,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        kernel_initializer="he_normal",
    )
)
model_1.add(
    Dense(
        2 * N,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        kernel_initializer="he_normal",
    )
)
model_1.add(Dense(1, activation="sigmoid"))


# In[ ]:

# Define a callback to modify the learning rate dynamically
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, min_lr=0.0001
)
# In[ ]:


# Compile Model
model_1.compile(
    optimizer="sgd",
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC", "Precision", "Recall"],
)
model_1.summary()


# In[ ]:


# Save the initial weights
initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
model_1.save_weights(initial_weights)


# In[ ]:

##DO ALL YOUR TRAINING USING MODEL_1###
start_time = time.time()
history_1 = model_1.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=300,
    validation_split=0.3,
    callbacks=[lr_callback],
)
end_time = time.time()
training_time = end_time - start_time


# In[ ]:


# Save Model
model_1.save("767Bands_100200300_model.h5")


# In[ ]:

loss, accuracy, auc, precision, recall = model_1.evaluate(X_test, y_test, verbose=0)
print("Test Loss: %.3f" % loss)
print("Test Accuracy: %.3f" % accuracy)
print("Test AUC: %.3f" % auc)
print("Test Precision: %.3f" % precision)
print("Test Recall: %.3f" % recall)
# print("Training time:", training_time, "seconds")

y_pred = model_1.predict(X_test)
y_pred = np.round(y_pred).tolist()
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

hist_df = pd.DataFrame(history_1.history)
cm_df = pd.DataFrame(cm)
test_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Test Loss",
            "Test Accuracy",
            "Test AUC",
            "Test Precision",
            "Test Recall",
            "Training Time",
        ],
        "Value": [loss, accuracy, auc, precision, recall, training_time],
    }
)

output_csv_file = "767Bands_100200300_Output.csv"

with open(output_csv_file, mode="w") as f:
    hist_df.to_csv(f, index=False)
    f.write("\n")
    cm_df.to_csv(f, index=False, header=False)
    f.write("\n")
    test_metrics_df.to_csv(f, index=False)


# In[ ]:


print(initial_weights)


# In[ ]:


##NOW ITS TIME TO RELOAD THE OLD WEIGHTS IN MODEL_1#


# In[ ]:


# This is where the old weights without training are being loaded
model_1.load_weights(initial_weights)


# In[ ]:


model_1.summary()
# The summary is same as model_1
# Our next goal is to modify the input shape.
# For doing this we will
# Then we will add a new layer which matching input dimensions that we need


# In[ ]:


# Model 2 -- 3 Feature Selection Algorithms
X, y = df.values[:, array_values(feature_array_triple) - 1], df.values[:, -1]
X = X.astype("float32")
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=17
)
n_features = X_train.shape[1]


# In[ ]:

# model_1 has the loaded weights.
# We are generating a new model_2 here.
# This is our final new model where we will copy the preliminary weights from model_1
model_2 = Sequential()
# create a new first layer with desired input_dimensions
model_2.add(
    Dense(
        N,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        kernel_initializer="he_normal",
        input_shape=(n_features,),
    )
)
# add layers with loaded weights sequentially
# go through all layers but the first one
for layer in model_1.layers[1:]:
    model_2.add(layer)


# In[ ]:


model_2.compile(
    optimizer="sgd",
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC", "Precision", "Recall"],
)
model_2.summary()


# In[ ]:

start_time = time.time()
history_2 = model_2.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=300,
    validation_split=0.3,
    callbacks=[lr_callback],
)
end_time = time.time()
training_time = end_time - start_time


# In[ ]:


model_2.save("3Feature_100200300_model.h5")


# In[ ]:

loss, accuracy, auc, precision, recall = model_2.evaluate(X_test, y_test, verbose=0)
print("Test Loss: %.3f" % loss)
print("Test Accuracy: %.3f" % accuracy)
print("Test AUC: %.3f" % auc)
print("Test Precision: %.3f" % precision)
print("Test Recall: %.3f" % recall)
print("Training time:", training_time, "seconds")

y_pred = model_2.predict(X_test)
y_pred = np.round(y_pred).tolist()
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

hist_df = pd.DataFrame(history_2.history)
cm_df = pd.DataFrame(cm)
test_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Test Loss",
            "Test Accuracy",
            "Test AUC",
            "Test Precision",
            "Test Recall",
            "Training Time",
        ],
        "Value": [loss, accuracy, auc, precision, recall, training_time],
    }
)

output_csv_file = "3Feature_100200300_Output.csv"

with open(output_csv_file, mode="w") as f:
    hist_df.to_csv(f, index=False)
    f.write("\n")
    cm_df.to_csv(f, index=False, header=False)
    f.write("\n")
    test_metrics_df.to_csv(f, index=False)


# In[ ]:


model_1.load_weights(initial_weights)
model_1.summary()


# In[ ]:


# Model 3 -- Literature Band Intersection
X, y = df.values[:, array_values(feature_array_LIT_intersect) - 1], df.values[:, -1]
X = X.astype("float32")
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=17
)
n_features = X_train.shape[1]


# In[ ]:

# model_1 has the loaded weights.
# We are generating a new model_3 here.
# This is our final new model where we will copy the preliminary weights from model_1
model_3 = Sequential()
# create a new first layer with desired input_dimensions
model_3.add(
    Dense(
        N,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        kernel_initializer="he_normal",
        input_shape=(n_features,),
    )
)
# add layers with loaded weights sequentially
# go through all layers but the first one
for layer in model_1.layers[1:]:
    model_3.add(layer)


# In[ ]:


model_3.compile(
    optimizer="sgd",
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC", "Precision", "Recall"],
)
model_3.summary()


# In[ ]:

start_time = time.time()
history_3 = model_3.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=300,
    validation_split=0.3,
    callbacks=[lr_callback],
)
end_time = time.time()
training_time = end_time - start_time


# In[ ]:


model_3.save("LitIntersection_100200300_model.h5")


# In[ ]:

loss, accuracy, auc, precision, recall = model_3.evaluate(X_test, y_test, verbose=0)
print("Test Loss: %.3f" % loss)
print("Test Accuracy: %.3f" % accuracy)
print("Test AUC: %.3f" % auc)
print("Test Precision: %.3f" % precision)
print("Test Recall: %.3f" % recall)
print("Training time:", training_time, "seconds")

y_pred = model_3.predict(X_test)
y_pred = np.round(y_pred).tolist()
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


hist_df = pd.DataFrame(history_3.history)
cm_df = pd.DataFrame(cm)
test_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Test Loss",
            "Test Accuracy",
            "Test AUC",
            "Test Precision",
            "Test Recall",
            "Training Time",
        ],
        "Value": [loss, accuracy, auc, precision, recall, training_time],
    }
)

output_csv_file = "LitIntersection_100200300_Output.csv"

with open(output_csv_file, mode="w") as f:
    hist_df.to_csv(f, index=False)
    f.write("\n")
    cm_df.to_csv(f, index=False, header=False)
    f.write("\n")
    test_metrics_df.to_csv(f, index=False)


# In[ ]:


model_1.load_weights(initial_weights)
model_1.summary()


# In[ ]:


# Model 4 -- Literature Bands 61
X, y = df.values[:, array_values(feature_array_LIT_61) - 1], df.values[:, -1]
X = X.astype("float32")
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=17
)
n_features = X_train.shape[1]


# In[ ]:

# model_1 has the loaded weights.
# We are generating a new model_4 here.
# This is our final new model where we will copy the preliminary weights from model_1
model_4 = Sequential()
# create a new first layer with desired input_dimensions
model_4.add(
    Dense(
        N,
        activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        kernel_initializer="he_normal",
        input_shape=(n_features,),
    )
)
# add layers with loaded weights sequentially
# go through all layers but the first one
for layer in model_1.layers[1:]:
    model_4.add(layer)


# In[ ]:


model_4.compile(
    optimizer="sgd",
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC", "Precision", "Recall"],
)
model_4.summary()


# In[ ]:

start_time = time.time()
history_4 = model_4.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=300,
    validation_split=0.3,
    callbacks=[lr_callback],
)
end_time = time.time()
training_time = end_time - start_time


# In[ ]:


model_4.save("LitBands61_100200300_model.h5")


# In[ ]:

loss, accuracy, auc, precision, recall = model_4.evaluate(X_test, y_test, verbose=0)
print("Test Loss: %.3f" % loss)
print("Test Accuracy: %.3f" % accuracy)
print("Test AUC: %.3f" % auc)
print("Test Precision: %.3f" % precision)
print("Test Recall: %.3f" % recall)
print("Training time:", training_time, "seconds")

y_pred = model_4.predict(X_test)
y_pred = np.round(y_pred).tolist()
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


hist_df = pd.DataFrame(history_4.history)
cm_df = pd.DataFrame(cm)
test_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Test Loss",
            "Test Accuracy",
            "Test AUC",
            "Test Precision",
            "Test Recall",
            "Training Time",
        ],
        "Value": [loss, accuracy, auc, precision, recall, training_time],
    }
)

output_csv_file = "LitBands61_100200300_Output.csv"

with open(output_csv_file, mode="w") as f:
    hist_df.to_csv(f, index=False)
    f.write("\n")
    cm_df.to_csv(f, index=False, header=False)
    f.write("\n")
    test_metrics_df.to_csv(f, index=False)
