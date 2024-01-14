import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPool1D, Bidirectional, LSTM, Concatenate, Dropout
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
import string
from tensorflow.keras.utils import plot_model
input_token = Input(shape = [], dtype = tf.string)
token_embedding = tf_hub_embedding_layer(input_token)
output_token= Dense(128, activation= "relu")(token_embedding)
sub_model_1 = Model(inputs = [input_token],
                     outputs = [output_token])
input_char = Input(shape=(1,), dtype= tf.string)
vectors = character_vectorizer(input_char)
char_embed = character_embedding(vectors)
bidirectional_lstm = Bidirectional(LSTM(25))(char_embed)
sub_model_2 = Model(inputs = [input_char],
                    outputs = [bidirectional_lstm], name = "sub_model_2")
#creating a hybrid embedding
hybrid_embedding= Concatenate(name= "hybrid_embedding")([sub_model_1.output,sub_model_2.output])
x = Dropout(0.5)(hybrid_embedding)
x = Dense(128, activation = "relu")(x)
x = Dropout(0.5)(x)
output = Dense(len(classes), activation = "softmax")(x)
Hybrid_model = Model(inputs = [sub_model_1.input, sub_model_2.input],
                     outputs = [output], name = "Hybrid_model")
Hybrid_model.summary()
