import datapreparation as dataprep
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy

import composer as c

def preprocess_data(train_data):
    X = []
    Y = []
    max_sequence_length = dataprep.get_max_length(train_data) - 1
    for j, song in enumerate(train_data):
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
        # Y_sample = np.vstack([Y_sample, np.ones(128)])


        X_sample = np.append(X_sample, np.zeros((max_sequence_length - X_sample.shape[0], 128)), axis=0)
        Y_sample = np.append(Y_sample, np.zeros((max_sequence_length - Y_sample.shape[0], 128)), axis=0)

        X.append(X_sample)
        Y.append(Y_sample)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def main():
    path = 'C:/Users/Phi Thien/PycharmProjects/TDT76/'

    training_data_f5 = dataprep.load_all_dataset('datasets/training/' + 'piano_roll_fs5', binarize=True)
    training_data_f2 = dataprep.load_all_dataset('datasets/training/' + 'piano_roll_fs2', binarize=True)
    training_data_f1 = dataprep.load_all_dataset('datasets/training/' + 'piano_roll_fs1', binarize=True)

    names_f1 = dataprep.load_all_dataset_names('datasets/training/' + 'piano_roll_fs1')
    names_f2 = dataprep.load_all_dataset_names('datasets/training/' + 'piano_roll_fs2')
    names = copy.deepcopy(names_f1)
    names.extend(names_f2)
    unique_names = sorted(list(set(names)))
    names_index = [[unique_names.index(name)] for name in names]


    training_data = copy.deepcopy(training_data_f1)


    training_data.extend(training_data_f2)
    # training_data.extend(training_data_f1)

    test = dataprep.test_piano_roll(training_data_f5[2], 15, fs=5)
    # dataprep.piano_roll_to_mid_file(training_data[0], 'test1.mid', fs=5)

    # dataprep.visualize_piano_roll(training_data[0], fs=5)

    path = 'generalist_lstm.h5'
    generalist = c.Generalist([128, 97, 128], len(training_data_f5), 'lstm', path)
    # generalist = c.Generalist([128, 43,43,43, 128], len(training_data_f5), 'gru')

    X, Y = preprocess_data(training_data)

    # generalist.train_network(10, X, Y)

    # generalist.save_network(path)
    generalist_result = generalist.gen_music(test.T, fs=5)

    dataprep.visualize_piano_roll(generalist_result, fs=5)
    dataprep.piano_roll_to_mid_file(generalist_result, 'gen_res1.mid', fs=5)

    # specialist = c.Specialist([128, 97, 128], len(training_data_f5), len(unique_names), generalist)








if __name__ == '__main__':
    main()