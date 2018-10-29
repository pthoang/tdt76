import tensorflow as tf
from tensorflow import keras

class Generalist:

    def __init__(self, dims, train):
        self.dims = dims
        self.train = train



    def build_network(self):

        model = keras.Sequential()


