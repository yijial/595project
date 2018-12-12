import sys
import numpy as np
import os
import json
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from scipy import stats
from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D
from keras.layers import LSTM,Input,Dense,Dropout,Activation
from keras.models import Model, Sequential
from keras.layers import Embedding

def main(seq_dir, cityname):
    file_dir = seq_dir+"/"+cityname+"/"
    embedding_index = {}
    f = open('glove.6B.300d.txt')
    for line in f:
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()

    print('found word vecs: ',len(embedding_index))

    x_test = np.load(file_dir+"x_test.npy")
    x_train = np.load(file_dir+"x_train.npy")
    y_test = np.load(file_dir+"y_test.npy")
    y_train = np.load(file_dir+"y_train.npy")
    bid_test = np.load(file_dir+"bid_test.npy")
    with open(file_dir+"word_index.json") as openfile:
        word_index = json.load(openfile)

    y_train_cat = to_categorical(np.asarray(y_train))
    y_test_cat = to_categorical(np.asarray(y_test))
    max_label_length = max(y_train_cat.shape[1], y_test_cat.shape[1])
    y_train_cat = pad_sequences(y_train_cat, maxlen=max_label_length, padding="post")
    y_test_cat = pad_sequences(y_test_cat, maxlen=max_label_length, padding="post")


    embedding_dim = 300
    max_seq_length = len(x_train[0])
    embedding_matrix = np.zeros((len(word_index)+1,embedding_dim))
    for word,i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    embedding_layer = Embedding(len(word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=max_seq_length,trainable=False)
    inp = Input(shape=(max_seq_length,))
    x = embedding_layer(inp)
    x = Bidirectional(LSTM(300,return_sequences=True,dropout=0.5,recurrent_dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(200,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(max_label_length,activation='linear')(x)
    x = Dense(max_label_length,activation='softmax')(x)
    model = Model(inputs=inp,outputs=x)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['mse'])
    model.fit(x_train,y_train_cat,validation_data=(x_test,y_test_cat),epochs=10,batch_size=100);
    output = model.predict(x_test)
    print(output.shape)

    predicted_count = np.argmax(output, axis=1)
    final_matrix = np.concatenate((y_test.reshape((y_test.shape[0],1)), predicted_count.reshape((predicted_count.shape[0],1)), bid_test.reshape((bid_test.shape[0],1))), axis = 1)
    filename = "prediction_lstm_class_"+cityname+".npy"
    np.save(filename, final_matrix)


if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
