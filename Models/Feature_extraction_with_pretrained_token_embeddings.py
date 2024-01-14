import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPool1D, Bidirectional, LSTM, Concatenate, Dropout
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
import string
from tensorflow.keras.utils import plot_model
input_data = Input(shape = [], dtype = tf.string)
pretrained_embedding = tf_hub_embedding_layer(input_data)
z = Dense(128, activation= "relu")(pretrained_embedding)
output_data = Dense(5, activation = "softmax")(z)
model_2 = Model(inputs = [input_data], outputs = [output_data])
