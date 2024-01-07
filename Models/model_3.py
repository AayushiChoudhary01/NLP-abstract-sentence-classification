input_data = Input(shape=(1,), dtype= tf.string)
vectors = character_vectorizer(input_data)
char_embed = character_embedding(vectors)
x = Conv1D(64, kernel_size= 5, activation= "relu", padding ="same")(char_embed)
x = GlobalMaxPool1D()(x)
output_data = Dense(len(classes), activation = "softmax")(x)
model_3 = Model(inputs = [input_data], outputs = [output_data], name = "model_3")
