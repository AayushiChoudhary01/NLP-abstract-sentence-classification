import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPool1D, Bidirectional, LSTM, Concatenate, Dropout
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Model
import string
from tensorflow.keras.utils import plot_model
input_token = Input(shape = [], dtype = "string", name = "inputs_token")
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
input_ln = Input(shape=(15,), dtype=tf.int32, name="input_ln")
z = Dense(32, activation="relu")(input_ln)
ln_feature_model = Model(inputs=[input_ln],
                                   outputs=[z])
input_total_line = Input(shape=(20,), dtype=tf.int32, name="input_total_line")
y = Dense(32, activation="relu")(input_total_line)
total_line_feature_model = tf.keras.Model(inputs=[input_total_line],
                                  outputs=[y])
token_char_embeddings = Concatenate(name="token_char_embeddings")([sub_model_1.output,
                                                                              sub_model_2.output])
x = Dense(256, activation="relu")(token_char_embeddings)
x = Dropout(0.5)(x)
x = Concatenate(name="token_char_positional_embedding")([ln_feature_model.output,
                                                                total_line_feature_model.output,
                                                                x])
output = Dense(5, activation="softmax", name="output")(x)
Tribid_model = Model(inputs=[ln_feature_model.input,
                        total_line_feature_model.input,
                        sub_model_1.input,
                        sub_model_2.input],
                outputs = [output], name ="Tribid_model")
print(Tribid_model.summary())
