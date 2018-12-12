import sys
import numpy as np
import os
import json
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D,MaxPooling1D,Flatten,BatchNormalization
from keras.layers import LSTM,Input,Dense,Dropout,Activation, SpatialDropout1D, Embedding
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras import backend as K

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1))))

def main(seq_dir, cityname):
    file_dir = seq_dir+"/"+cityname+"/"
    x_test = np.load(file_dir+"x_test.npy")
    x_train = np.load(file_dir+"x_train.npy")
    y_test = np.load(file_dir+"y_test.npy")
    y_train = np.load(file_dir+"y_train.npy")
    bid_test = np.load(file_dir+"bid_test.npy")
    with open(file_dir+"word_index.json") as openfile:
        word_index = json.load(openfile)
    embedding_index = {}

    f = open('glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()

    print('found word vecs: ',len(embedding_index))

    max_seq_length = x_train.shape[1]
    embedding_dim = 300
    embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
    embedding_matrix.shape

    for word,i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length,trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.1))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Bidirectional(LSTM(300,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))


    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    rm = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
    model.compile(optimizer=sgd,loss=rmsle,metrics=['accuracy'])

    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=1000)

    output = model.predict(x_test)

    predicted_count = output
    true_count = y_test
    final_matrix = np.concatenate((true_count.reshape((len(true_count),1)), predicted_count.reshape((len(predicted_count),1)), bid_test.reshape((len(predicted_count),1))), axis = 1)

    for i in range(len(final_matrix)):
        print(final_matrix[i])
    filename = "prediction_lstm_reg_"+cityname+".npy"
    np.save(filename, final_matrix)

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])