from numpy import zeros, asarray
from keras.models import load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras import Sequential
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint

import data_cleaning as dcl

def multiprocess_func(data_list, model):
    predictions = []
    for sentence in data_list:
        predictions.append(model.predict(sentence))
    return predictions


# @dcl.timer
class NNModelHandler:
    
    def __init__(self, train_data_cleaner, pred_data_cleaner, embedding_file_loc, save_model_to_location='', load_model_from_location='', num_of_epochs=25, verbose=1, vocab_size=10000, batch_size=32):
        self.train_data_cleaner = train_data_cleaner
        self.pred_data_cleaner = pred_data_cleaner
        self.vocab_size = vocab_size
        self.num_of_epochs = num_of_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.save_model_to_location = save_model_to_location
        self.load_model_from_location = load_model_from_location
        
        if self.load_model_from_location:
            self.model = load_model(self.load_model_from_location)
            
        else:
            self.embedding_matrix = self._get_embedding_matrix(embedding_file_loc)
            self.model = self._compile_NN_model()
            self.model.train_NN_model()
    
    
    # @dcl.timer
    def train_NN_model(self):
        # train_index, test_index = dcl.train_test_split_index_for_generators(self.train_data_cleaner.train_data_files, test_size=0.2)
        
        checkpoint = ModelCheckpoint(
                            self.save_model_to_location,
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            verbose=1)
        
        # TEMPORARY SOLUTION - until I solve the issue Windows has with generators and fitting, and Linux doesn't
        data_gen = self.train_data_cleaner.get_clean_train_data_generator(self.train_data_cleaner.train_data_files, as_sequences=True, sentiment_as_one_hot=True)
        
        X_data = DataFrame()
        y_data = DataFrame()
        for entry in data_gen:
            X_data = X_data.append(DataFrame(entry[0]), ignore_index=True)
            y_data = y_data.append(DataFrame(entry[1]), ignore_index=True)
    
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
        
        # this is planned to work with generators in the future (maybe I'll just install Linux or a subsystem, or will figure out how to make generators thread-safe properly)
        # for the time being this will accept memory-consuming pandas dataframes
        self.model.fit(x=X_train, y=y_train, epochs=self.num_of_epochs, validation_data=(X_test, y_test), verbose=self.verbose, batch_size=self.batch_size, callbacks=[checkpoint])#, workers=1)
        
    # @dcl.timer
    def _compile_NN_model(self, loss='categorical_crossentropy', optimizer='Adam', output_activation='softmax', metrics=['accuracy']):
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=100, weights = [self.embedding_matrix], input_length = self.pred_data_cleaner.max_length, trainable=False),
            Bidirectional(LSTM(64, input_shape=(1,100), return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
            Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5)),
            Dense(3, activation=output_activation)])
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        return model        
    
    # @dcl.timer
    def _get_embedding_matrix(self, embedding_file_loc):
        # vocab_size = len(self.text_cleaner.tokenizer.word_index,)+1
        vocab_size = self.vocab_size

        embeddings_index={}
        with open(embedding_file_loc, 'r', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = zeros((vocab_size, 100))

        for word, i in self.pred_data_cleaner.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe
            if embedding_vector is not None:
                # add to matrix
                embedding_matrix[i] = embedding_vector # each row of matrix
        return embedding_matrix
    
    
    

