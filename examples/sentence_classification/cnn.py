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


def create_cnn_sent(input_vdim, num_class, vocab_size, embedding_dim):
	inputs = keras.Input(shape=(input_vdim,))
	embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
	embedded_x = keras.layers.Embedding(vocab_size+1, embedding_dim, embeddings_initializer=embed_initer)(inputs)
	embedded_x = keras.layers.Reshape((embedded_x[0].shape.concatenate(1)))(embedded_x)

	filters_outputs=[]

	
	if input_vdim <= 2:
		filter_sizes=[i for i in range(1, input_vdim+1)]
	elif input_vdim < 5 and input_vdim > 2:
		filter_sizes=[i for i in range(2, input_vdim+1)]
	else:
		filter_sizes=[3,4,5]
	
	for filter_size in filter_sizes:
		conv=keras.layers.Conv2D(100, (filter_size, embedding_dim), activation='relu', kernel_initializer='glorot_normal', name='convolution_{:d}'.format(filter_size))(embedded_x)
		pooled=keras.layers.MaxPool2D(pool_size=(embedded_x.shape[1]-filter_size+1, 1), strides=(1, 1), padding='valid')(conv)
		filters_outputs.append(pooled)

	filters_outputs=keras.layers.concatenate(filters_outputs)
	filters_outputs = keras.layers.Flatten(name='flatten')(filters_outputs)
	filters_outputs=keras.layers.Dropout(0.5)(filters_outputs)

	output=keras.layers.Dense(num_class, activation='softmax', kernel_initializer='glorot_normal', 
		kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01))(filters_outputs)

	return keras.Model(inputs=inputs, outputs=output)

def train_cnn_sent(train_x, train_y, embedding_dim, epoch, num_words=None):

	vector_x, tokenizer=load_data(train_x, num_words)
	vector_y = one_hot(train_y, len(set(train_y)))
	
	if not num_words:
		num_words=len(tokenizer.word_index)

	cnn_model=create_cnn_sent(len(vector_x[0]), len(set(train_y)), num_words, embedding_dim)
	cnn_model.compile(optimizers.Adadelta(), loss='binary_crossentropy',#loss='categorical_crossentropy',
                           metrics=['accuracy'])

	print(cnn_model.summary())

	cnn_model.fit(x=vector_x, y=vector_y, batch_size=128, epochs=epoch, validation_split=0.1, shuffle=True)

	return cnn_model, tokenizer, len(vector_x[0])

def predict_cnn_sent(model, x_test, tokenizer, max_char_leng):
	vector_x=load_data(x_test, predict=True, tokenizer=tokenizer, max_char_leng=max_char_leng)
	results=model.predict(x=vector_x, batch_size=1, verbose=1)
	results=tf.math.argmax(results, axis=-1)
	return tf.reshape(results, [results.shape[0]])