import keras
from keras import layers
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

def encode_input(input_img):
    encoding_dim = 32
    input_img = keras.Input(shape=(784,))
    
    #initialize encoder model
    encoded=layers.Dense(encoding_dim,activation='relu')(input_img)
    decoded=layers.Dense(784,activation='sigmoid')(encoded)

    autoencoder=keras.Model(input_img)
    
    #encoder section
    encoder=keras.Model(input_img,encoded)
    encoded_input = keras.Input(input_img,encoded)


    #decoder section
    encoded_input=keras.Input(shape=(encoding_dim,))
    decoder_layer=autoencoder.layers[-1]
    decoder=keras.Model(encoded_input,decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam',loss='binary_crossentropy')


def encode():
    pass

def get_distance():
    pass

def decode():
    pass

