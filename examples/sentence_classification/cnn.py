import argparse
import os

from tensorflow import one_hot, keras, optimizers
import tensorflow as tf


def read_txt(file_path, labels=False):
	"""Returns a list object from a text file
		
		:param file_path: the path of the input file
		:type file_path: str
		:param labels: indicates whether the input_file contains labels (used for type conversion), defaults to False
		:type labels: boolean
        
        :return: list object read from the text file
        :rtype: list
	"""
	list_elem=[]
	with open(file_path, 'r') as f:
		for line in f:
			if labels==True:
				list_elem.append(int(line.rstrip()))
			else:
				list_elem.append(line.rstrip())
	return list_elem

def load_data(input_data, num_words=None, predict=False, tokenizer=None, max_word_len=0):
	"""Returns a numpy array of the indexed input data and Python dictionary with tensorflow tokenized information
	   If predict is True, return only a numpy array of the indexed input data
		
		:param input_data: a list of documents (each document as a list of tokens or words)
		:type input_data: list of list
		:param num_words: top N words to be used in the model, defaults to None
		:type num_words: int, optional
		:param predict: indicator for distinguishing training or inference data, defaults to None
		:type predict: boolean
		:param tokenizer: tokenization information to be used for loading data during inference, defaults to None
		:type tokenizer: dict
		:param max_word_len: maximum number of words in a document vector used for padding vectors during inference, defaults to 0
		:type max_word_len: int
        
        if predict = True
        :return: a numpy array of the document vectors for inference, indexed by the input tokenizer dictionary
        :rtype: numpy array
        if predict = False
        :return: a numpy array of document vectors, indexed by the tokenizer dictionary
        :rtype: numpy array, dict
	"""
	if predict:
		tokenizer.fit_on_texts(input_data)
		tensor=tokenizer.texts_to_sequences(input_data)
		tensor=keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_word_len, padding='post')
		return tensor

	lang_tokenizer=keras.preprocessing.text.Tokenizer(filters='', num_words=num_words, oov_token='UNK')
	lang_tokenizer.fit_on_texts(input_data)

	tensor=lang_tokenizer.texts_to_sequences(input_data)
	tensor=keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

	return tensor, lang_tokenizer

def _cal_filter_sizes(input_vdim):
	"""Returns a list of filter sizes (size of n-grams) to be used.
		For accommodating input tensors with size less than 5
		(Original Yoon's model assume the minimum size of the input tensor to be bigger than 5) 
		
		:param input_vdim: maximum dimension of the input tensor
		:type input_vdim: int
        
        :return: a list of filter sizes
        :rtype: list
	"""
	if input_vdim<=2:
		filter_sizes=[i for i in range(1, input_vdim+1)]
	elif input_vdim<5 and input_vdim>2:
		filter_sizes=[i for i in range(2, input_vdim+1)]
	else:
		filter_sizes=[3,4,5]
	return filter_sizes

def create_cnn_sent(input_vdim, num_class, vocab_size, embedding_dim):
	"""Returns Yoon Kim's sentence classification model
		Model architecture and hyperparameter set as described in "Convolutional neural networks for sentence classification"
		
		:param input_vdim: maximum dimension of the input tensor
		:type input_vdim: int
		:param num_class: the unique number of classes to predict
		:type num_class: int
		:param vocab_size: the unique number of words to be embedded
		:type vocab_size: int
		:param embedding_dim: the dimension of embedding word vectors
		:type embedding_dim: int
        
        :return: Yoon Kim's sentence classification CNN model
        :rtype: Keras Model object
	"""
	inputs=keras.Input(shape=(input_vdim,))
	embed_initer=keras.initializers.RandomUniform(minval=-1, maxval=1)
	embedded_x=keras.layers.Embedding(vocab_size+1, embedding_dim, embeddings_initializer=embed_initer)(inputs)
	embedded_x=keras.layers.Reshape((embedded_x[0].shape.concatenate(1)))(embedded_x)

	filters_outputs=[]
	
	for filter_size in _cal_filter_sizes(input_vdim):
		conv=keras.layers.Conv2D(100, (filter_size, embedding_dim), activation='relu', kernel_initializer='glorot_normal', name='convolution_{:d}'.format(filter_size))(embedded_x)
		pooled=keras.layers.MaxPool2D(pool_size=(embedded_x.shape[1]-filter_size+1, 1), strides=(1, 1), padding='valid')(conv)
		filters_outputs.append(pooled)

	filters_outputs=keras.layers.concatenate(filters_outputs)
	filters_outputs=keras.layers.Flatten(name='flatten')(filters_outputs)
	filters_outputs=keras.layers.Dropout(0.5)(filters_outputs)

	output=keras.layers.Dense(num_class, activation='softmax', kernel_initializer='glorot_normal', 
		kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01))(filters_outputs)

	return keras.Model(inputs=inputs, outputs=output)

def train_cnn_sent(train_x, train_y, embedding_dim, epoch, num_words=None):
	"""Returns a trained CNN sentence classification model along with its tokenizer and dimension of the sentence vector
		
		:param train_x: a list of documents (each document as a list of tokens or words)
		:type train_x: list of list
		:param train_y: a list of labels
		:type train_y: list
		:param embedding_dim: the dimension of embedding word vectors
		:type embedding_dim: int
		:param epoch: the number of training epochs
		:type epoch: int
		:param num_words: top N words to be used in the model, defaults to None
		:type num_words: int, optional
        
        :return: Yoon Kim's sentence classification CNN trained model, tokenizer used in the model, dimension of its document vectors
        :rtype: Keras Model object, dict, int
	"""
	vector_x, tokenizer=load_data(train_x, num_words)
	vector_y=one_hot(train_y, len(set(train_y)))
	
	if not num_words:
		num_words=len(tokenizer.word_index)

	cnn_model=create_cnn_sent(len(vector_x[0]), len(set(train_y)), num_words, embedding_dim)
	cnn_model.compile(optimizers.Adadelta(learning_rate=1.0), loss='binary_crossentropy',#loss='categorical_crossentropy',
                           metrics=['accuracy'])

	print(cnn_model.summary())

	cnn_model.fit(x=vector_x, y=vector_y, batch_size=128, epochs=epoch, validation_split=0.1, shuffle=True)

	return cnn_model, tokenizer, len(vector_x[0])

def predict_cnn_sent(model, x_test, tokenizer, max_word_len):
	"""Returns a list of predicted labels of x_test, computed from the input model
		
		:param model: a trained CNN sentence classification model
		:type model: Keras Model object
		:param x_test: a list of documents (each document as a list of tokens or words)
		:type x_test: list of list
		:param tokenizer: word tokenizer used during training (for indexing the test vectors)
		:type tokenizer: dict
		:param max_word_len: maximum number of words in a document vector used for padding vectors during inference 
		:type max_word_len: int
        :return: a list of predicted labels
        :rtype: tf.Tensor
	"""
	vector_x=load_data(x_test, predict=True, tokenizer=tokenizer, max_word_len=max_word_len)
	results=model.predict(x=vector_x, batch_size=1, verbose=1)
	results=tf.math.argmax(results, axis=-1)
	return tf.reshape(results, [results.shape[0]])


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CNN model for sentence classification')
	parser.add_argument('-x', '--train_x', type=str, help='Path for the input corpus')
	parser.add_argument('-y', '--train_y', type=str, help='Path for the labels')
	parser.add_argument('-d', '--dim', default=200, type=int, help='Dimensions for the embedding vectors')
	parser.add_argument('-e', '--epoch', default=10, type=int, help='Number of training epochs')
	parser.add_argument('-w', '--num_words', default=None, type=int, help='Number of words to be used for training')
	args = parser.parse_args()
	print('Parameters:', args, '\n')

	train_x=read_txt(args.train_x)
	train_y=read_txt(args.train_y, labels=True)

	model, tokenizer, max_sen_length=train_cnn_sent(train_x, train_y, args.dim, args.epoch, args.num_words)
