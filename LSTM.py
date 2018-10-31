import tensorflow as tf
from tensorflow import keras
import datapreparation as dataprep
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from sklearn import preprocessing

class Generalist:

    def __init__(self, dims, train_data, saved_model=None):
        self.dims = dims
        self.train_data = train_data

        self.model = keras.models.load_model(saved_model) if saved_model else self.build_network()
        if not saved_model: self.init_network()

        print(self.model.summary())


    def build_network(self):

        model = keras.Sequential()

        model.add(keras.layers.LSTM(self.dims[1], return_sequences=True, stateful=False, batch_input_shape=(len(self.train_data),None, 128)))
        for dim in self.dims[2:-1]:
            model.add(keras.layers.LSTM(dim, return_sequences=True, stateful=False))
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(self.dims[len(self.dims)-1], activation='linear'))


        return model

    def init_network(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01),
            loss='mse'
        )

    def train_network(self, epochs):
        X = []
        Y = []
        max_sequence_length = dataprep.get_max_length(self.train_data) - 1
        for j, song in enumerate(self.train_data):
            # min_max_scaler_X = preprocessing.MinMaxScaler()
            # min_max_scaler_Y = preprocessing.MinMaxScaler()
            # X = song[:, 1:].T
            # Y = X[1:]
            # X = np.vstack([X, np.full(128, 127)])
            # X = np.vstack((X, np.zeros(128)))
            # Y = np.vstack([Y, np.full(128, 127)])
            # Y = np.vstack((Y, np.zeros(128)))
            #
            # X = min_max_scaler_X.fit_transform(X)
            # Y = min_max_scaler_Y.fit_transform(Y)
            #
            # X = X[:-2]
            # Y = Y[:-1]

            X_sample = song[:, 1:].T
            Y_sample = X_sample[1:]
            Y_sample = np.vstack([Y_sample, np.ones(128)])


            X_sample = np.append(X_sample, np.zeros((max_sequence_length - X_sample.shape[0], 128)), axis=0)
            Y_sample = np.append(Y_sample, np.zeros((max_sequence_length - Y_sample.shape[0], 128)), axis=0)


            X.append(X_sample)
            Y.append(Y_sample)

        X = np.array(X)
        Y = np.array(Y)

        print(X.shape)

        self.model.fit(X, Y, epochs=epochs, batch_size=len(self.train_data))

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
        return len(list(filter(lambda x: x > 0.1, step))) == 128

    def build_gen_music_model(self):
        model = keras.Sequential()

        model.add(keras.layers.LSTM(self.dims[1], return_sequences=True, stateful=True,
                                    batch_input_shape=(1, None, 128)))
        for dim in self.dims[2:-1]:
            model.add(keras.layers.LSTM(dim, return_sequences=True, stateful=True))
            model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(self.dims[len(self.dims) - 1], activation='linear'))

        old_weights = self.model.get_weights()

        model.set_weights(old_weights)
        return model

    def gen_music(self, init, fs=5):
        result = []

        model = self.build_gen_music_model()

        print(init)

        pred = model.predict(init.reshape(1, init.shape[0], init.shape[1]))



        for step in pred[0]:
            max = np.amax(step)
            min = np.amin(step)
            step = [(x - min)/(max-min) for x in step]
            if(self.check_end_of_song(step)):
                break
            new_step = [1 if x > 0.5 else 0 for x in step]
            result.append(new_step)

        for i in range(100*fs):
            step = model.predict(np.array(result[-1:]).reshape(1, 1, 128))[0][0]
            max = np.amax(step)
            min = np.amin(step)
            step = [(x - min) / (max - min) for x in step]
            if(self.check_end_of_song(step)):
                break
            new_step = [1 if x > 0.5 else 0 for x in step]
            result.append(new_step)


        return np.array(result).T




