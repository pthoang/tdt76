import tensorflow as tf
from tensorflow import keras
import datapreparation as dataprep
import numpy as np
import torch
import math

from sklearn import preprocessing

class Generalist:

    def __init__(self, dims, batch_size, type, saved_model=None):
        self.dims = dims
        self.batch_size = batch_size
        self.type = type

        self.model = keras.models.load_model(saved_model) if saved_model else self.build_network()
        if not saved_model: self.init_network()

        print(self.model.summary())


    def build_network(self):

        model = keras.Sequential()
        model.add(keras.layers.Masking(mask_value=0., batch_input_shape=(self.batch_size,None, 128)))
        if self.type == 'lstm':
            for dim in self.dims[1:-1]:
                model.add(keras.layers.LSTM(dim, return_sequences=True, stateful=False))
                model.add(keras.layers.Dropout(0.2))

        elif self.type == 'gru':
            for dim in self.dims[1:-1]:
                model.add(keras.layers.GRU(dim, return_sequences=True, stateful=False))
                model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(self.dims[len(self.dims)-1], activation='relu'))

        for layer in model.layers:
            print(layer)
            print(layer.get_weights())
        return model

    def init_network(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.003),
            loss='mse'
        )

    def train_network(self, epochs, X, Y):

        self.model.fit(X, Y, epochs=epochs, batch_size=self.batch_size)

        # for i in range(epochs):
        #     print("Epoch number: " + str(i+1))
        #
        #     # self.model.fit(X.reshape(1, X.shape[0], X.shape[1]), Y.reshape(1, Y.shape[0], Y.shape[1]), epochs=1, batch_size=1, shuffle=False)
        #
        #     self.model.reset_states()


            # pred = self.model.predict(self.train_data[0].T.reshape(1, self.train_data[0].T.shape[0], self.train_data[0].T.shape[1]))
            #
            # result = []
            # for line in pred[0]:
            #     new_line = [1 if x > 0.15 else 0 for x in line]
            #     result.append(new_line)
            #
            # dataprep.visualize_piano_roll(np.array(result).T, fs=5)
            #
            # dataprep.piano_roll_to_mid_file(np.array(result).T,'res1.mid', fs=5)



    def save_network(self, path):
        self.model.save(path)

    def check_end_of_song(self, step):
        return len(list(filter(lambda x: x > 0.5, step))) == 128

    def build_gen_music_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Masking(mask_value=0., batch_input_shape=(1, None, 128)))
        if self.type == 'lstm':
            for dim in self.dims[1:-1]:
                model.add(keras.layers.LSTM(dim, return_sequences=True, stateful=True))
                model.add(keras.layers.Dropout(0.2))

        elif self.type == 'gru':
            for dim in self.dims[1:-1]:
                model.add(keras.layers.GRU(dim, return_sequences=True, stateful=True))
                model.add(keras.layers.Dropout(0.2))


        model.add(keras.layers.Dense(self.dims[len(self.dims) - 1], activation='relu'))

        old_weights = self.model.get_weights()

        model.set_weights(old_weights)
        return model

    def gen_music(self, init, fs=5):
        result = []

        model = self.build_gen_music_model()

        pred = model.predict(init.reshape(1, init.shape[0], init.shape[1]))



        for step in pred[0]:
            # max = np.amax(step)
            # min = np.amin(step)
            # step = [(x - min)/(max-min) for x in step]
            if(self.check_end_of_song(step)):
                print(step)
                break
            new_step = [1 if x > 0.2 else 0 for x in step]
            result.append(new_step)

        # model.reset_states()
        # X = init
        # X = np.append(X, result[-1:], axis=0)
        for i in range(100*fs):
            # step = model.predict(X.reshape(1, X.shape[0], X.shape[1]))[0][-1]
            step = model.predict(np.array([result[-1:]]))[0][0]
            # max = np.amax(step)
            # min = np.amin(step)
            # step = [(x - min) / (max - min) for x in step]
            if(self.check_end_of_song(step)):
                break
            new_step = [1 if x > 0.2 else 0 for x in step]
            result.append(new_step)
            # X = np.append(X, result[-1:], axis=0)

        model.reset_states()



        return np.array(result).T


class SpecialistModel(tf.keras.Model):

    def __init__(self, generalist_dims, batch_size, gen_music=False):
        super(SpecialistModel, self).__init__()
        self.batch_size = batch_size

        layer_batch_size = 1 if gen_music else self.batch_size



        self.decoder_input = keras.layers.Masking(mask_value=0., batch_input_shape=(layer_batch_size,None, 128))
        self.decoder_lstm = []
        self.decoder_dropout = []
        for dim in generalist_dims[1:-1]:
            self.decoder_lstm.append(keras.layers.LSTM(dim, return_sequences=True, stateful=gen_music))
            self.decoder_dropout.append(keras.layers.Dropout(0.2))
        self.decoder_output = keras.layers.Dense(generalist_dims[-1], activation=('sigmoid' if gen_music else 'relu'))
        self.tags_embedding = keras.layers.Embedding(2, len(self.decoder_lstm))


    def call(self, inputs, hidden=None, training=False):
        print(inputs)
        x = self.decoder_input(inputs)
        for i in range(len(self.decoder_lstm)):
            x = self.decoder_lstm[i](x)
            if training:
                x = self.decoder_dropout[i](x, training=training)

        print(hidden)
        return self.decoder_output(x), hidden


class Specialist:

    def __init__(self, generalist_dims, batch_size, num_tags, generalist, saved_model=None):
        self.gen_dims = generalist_dims
        self.batch_size = batch_size
        self.num_tags = num_tags
        self.generalist = generalist.model

        self.model = self.build_network()


    def build_network(self, gen_music=False):

        encoder_inputs = keras.layers.Input(shape=(1,1))

        decoder_inputs = keras.layers.Input(shape=(None, 128))

        decoder_input = keras.layers.Masking(mask_value=0., batch_input_shape=(self.batch_size, None, 128))
        # decoder_input.set_weights(self.generalist.layers[0].get_weights())
        decoder_gru = keras.layers.LSTM(self.gen_dims[1], return_sequences=True, stateful=gen_music)
        decoder_dropout = keras.layers.Dropout(0.2)


        decoder_output = keras.layers.Dense(self.gen_dims[-1], activation='sigmoid' if gen_music else 'relu')

        # decoder_output.set_weights(self.generalist.layers[-1].get_weights())

        encoder = keras.layers.LSTM(self.gen_dims[1], return_sequences=True, return_state=True)

        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        init_state = [state_h, state_c]

        output = decoder_input(decoder_inputs)
        # print(self.num_tags)

        output = decoder_gru(output, initial_state=init_state)
        output = decoder_dropout(output)

        output = decoder_output(output)

        model = keras.Model([encoder_inputs, decoder_inputs], output)

        for i, layer in enumerate(model.layers[1:]):
            print(layer)
            print(self.generalist.layers[i])
            print(layer.get_weights())
            layer.set_weights(self.generalist.layers[i].get_weights())


        return model








