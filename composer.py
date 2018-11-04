import tensorflow as tf
from tensorflow import keras
import datapreparation as dataprep
import numpy as np
import torch
import math

from sklearn import preprocessing


def check_end_of_song(step):
    return len(list(filter(lambda x: x > 0.1, step))) == 128

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
            if(check_end_of_song(step)):
                print(step)
                break
            new_step = [1 if x > 0.15 else 0 for x in step]
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
            if(check_end_of_song(step)):
                break
            new_step = [1 if x > 0.15 else 0 for x in step]
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
        self.decoder_output = keras.layers.Dense(generalist_dims[-1], activation='relu')
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


        self.model =  self.build_network()
        # if saved_model:
        #     interim_model = self.build_network()
        #     interim_model.load_weights(saved_model)
        #     layers = [0,1,2,3,5]
        #     for i in layers:
        #         self.model.layers[i].set_weights(interim_model.layers[i].get_weights())

        if saved_model:
            self.model.load_weights(saved_model)

        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss='mse'
        )


        print(self.model.summary())

    def custom(self, input, gen_music):
        # input = tf.gather(input, [0,2])


        size = 1 if gen_music else self.batch_size
        input = tf.reshape(input, [size, self.gen_dims[1]])
        return input


    def build_network(self, gen_music=False):


        encoder_inputs = keras.layers.Input(shape=(1,1))

        decoder_inputs = keras.layers.Input(shape=(None, 128))

        encoder_embed_h = keras.layers.Embedding(self.num_tags+1, self.gen_dims[1])
        encoder_reshape = keras.layers.Reshape((1,self.gen_dims[1]), input_shape=(None, 1, 1, 97))
        # encoder_embed_lambda = keras.layers.Lambda(lambda input: tf.reshape(input, [1,self.gen_dims[1]]))
        encoder_embed_lambda = keras.layers.Lambda(lambda input: self.custom(input, gen_music))
        state_h = encoder_embed_h(encoder_inputs)
        state_h = encoder_embed_lambda(state_h)

        encoder_embed_c = keras.layers.Embedding(self.num_tags + 1, self.gen_dims[1])
        state_c = encoder_embed_c(encoder_inputs)
        state_c = encoder_embed_lambda(state_c)

        init_state = [state_h, state_c]

        decoder_input_layer = keras.layers.Masking(mask_value=0.,
                                             batch_input_shape=(1 if gen_music else
                                                                self.batch_size, None, 128))
        # decoder_input.set_weights(self.generalist.layers[0].get_weights())
        decoder_lstm = keras.layers.LSTM(self.gen_dims[1], return_sequences=True,
                                         return_state=True,
                                         stateful=False)
        decoder_dropout = keras.layers.Dropout(0.2)


        decoder_output = keras.layers.Dense(self.gen_dims[-1], activation='relu')

        # decoder_output.set_weights(self.generalist.layers[-1].get_weights())

        # encoder = keras.layers.LSTM(self.gen_dims[1], return_sequences=True, return_state=True)

        # encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        output = decoder_input_layer(decoder_inputs)
        # print(self.num_tags)

        output, _, _ = decoder_lstm(output, initial_state=init_state)
        output = decoder_dropout(output)

        output = decoder_output(output)

        model = keras.Model([decoder_inputs, encoder_inputs], output)

        for i, layer in enumerate(model.layers):
            print(i, layer)


        copy_weight_indexes = [4,6,7,8]

        for i, layer_i in enumerate(copy_weight_indexes):
            model.layers[layer_i].set_weights(self.generalist.layers[i].get_weights())




        # inference models
        # encoder_model = keras.Model(encoder_inputs, init_state)
        #
        # decoder_state_h = keras.Input(shape=(self.gen_dims[1],))
        # decoder_state_c = keras.Input(shape=(self.gen_dims[1],))
        # decoder_init_state = [decoder_state_h, decoder_state_c]
        #
        # decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
        #                                                  initial_state=decoder_init_state
        #                                                  )
        #
        # decoder_state = [state_h, state_c]
        #
        # decoder_outputs = decoder_output(decoder_outputs)
        #
        # decoder_model = keras.Model([decoder_inputs] + decoder_init_state,
        #                             [decoder_outputs] + decoder_state)
        # # decoder_model = keras.Model(decoder_inputs,
        # #                             decoder_outputs)
        #
        #
        # return model, encoder_model, decoder_model

        return model

    def save_network(self, path):
        self.model.save_weights(path)

    def train_network(self, epochs, X_decoder, Y_decoder, X_encoder):


        self.model.fit([X_decoder, X_encoder], Y_decoder, epochs=epochs, batch_size=self.batch_size)

    def gen_music(self, init, composer, fs=5):

        result = []

        model = self.build_network(gen_music=True)
        model.set_weights(self.model.get_weights())

        pred = model.predict([init.reshape(1, init.shape[0], init.shape[1]), composer])

        for step in pred[0]:
            if(check_end_of_song(step)):
                print(step)
                break
            new_step = [1 if x > 0.15 else 0 for x in step]
            result.append(new_step)

        X = init
        X = np.append(X, result[-1:], axis=0)
        for i in range(100 * fs):
            step = model.predict([X.reshape(1, X.shape[0], X.shape[1]), composer])[0][-1]
            if (check_end_of_song(step)):
                break
            new_step = [1 if x > 0.15 else 0 for x in step]
            result.append(new_step)
            X = np.append(X, result[-1:], axis=0)

        # state = self.encoder_model.predict(composer)
        #
        # print(state)
        #
        # steps = len(init) + 100*fs + 1
        #
        # for step in range(steps):
        #     outputs, h, c = self.decoder_model.predict([init.reshape(1, init.shape[0], init.shape[1])]
        #                                                + state)
        #     # outputs = self.decoder_model.predict(init.reshape(1, init.shape[0], init.shape[1]))
        #     if step == 0:
        #         for output in outputs[0]:
        #             new_output = [1 if x > 0 else 0 for x in output]
        #             result.append(new_output)
        #
        #     else:
        #         new_output = [1 if x > 0 else 0 for x in outputs[0][-1]]
        #
        #         result.append(new_output)
        #         init = np.append(init, result[-1:], axis=0)
        #     state = [h, c]


        return np.array(result).T








