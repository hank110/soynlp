from tensorflow import one_hot, keras, optimizers
import tensorflow as tf

def load_data(input_data, num_words=None, predict=False, tokenizer=None, max_char_leng=0):
	if predict:
		tokenizer.fit_on_texts(input_data)
		tensor=tokenizer.texts_to_sequences(input_data)
		tensor = keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_char_leng, padding='post')
		return tensor

	lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='', num_words=num_words, oov_token='UNK')
	lang_tokenizer.fit_on_texts(input_data)

	tensor = lang_tokenizer.texts_to_sequences(input_data)
	tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

	return tensor, lang_tokenizer

def create_lstm_sent(input_vdim, num_class, vocab_size, embedding_dim, lstm_hidden):
	inputs = keras.Input(shape=(input_vdim,))
	embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
	embedded_x = keras.layers.Embedding(vocab_size+1, embedding_dim)(inputs)
	
	lstm=keras.layers.Bidirectional(keras.layers.LSTM(lstm_hidden,return_sequences=True))(embedded_x)
	lstm=keras.layers.Bidirectional(keras.layers.LSTM(lstm_hidden,return_sequences=True))(lstm)
	bi_lstm=keras.layers.Bidirectional(keras.layers.LSTM(lstm_hidden))(lstm)

	output=keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=keras.regularizers.l2(3), bias_regularizer=keras.regularizers.l2(3))(bi_lstm)

	return keras.Model(inputs=inputs, outputs=output)

def train_lstm_sent(train_x, train_y, embedding_dim, lstm_hidden, epoch, num_words=None):
	vector_x, tokenizer=load_data(train_x, num_words)
	vector_y = one_hot(train_y, len(set(train_y)))

	if not num_words:
		num_words=len(tokenizer.word_index)
	
	lstm_model=create_lstm_sent(len(vector_x[0]), len(set(train_y)), num_words, embedding_dim, lstm_hidden)
	lstm_model.compile(optimizers.Adadelta(), loss='categorical_crossentropy',
                           metrics=['accuracy'])

	print(lstm_model.summary())

	lstm_model.fit(x=vector_x, y=vector_y, batch_size=1, epochs=epoch, validation_split=0.1, shuffle=True)

	return lstm_model, tokenizer, len(vector_x[0])

def predict_lstm_sent(model, x_test, tokenizer, max_char_leng):
	vector_x=load_data(x_test, predict=True, tokenizer=tokenizer, max_char_leng=max_char_leng)
	results=model.predict(x=vector_x, batch_size=1, verbose=1)
	results=tf.math.argmax(results, axis=-1)
	return tf.reshape(results, [results.shape[0]])