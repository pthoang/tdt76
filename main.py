import datapreparation as dataprep
import numpy as np

from LSTM import Generalist

def main():
    path = 'C:/Users/Phi Thien/PycharmProjects/TDT76/'
    dataset = 'piano_roll_fs5'

    training_data = dataprep.load_all_dataset('datasets/training/' + dataset, binarize=True)

    names = dataprep.load_all_dataset_names('datasets/training/' + dataset)


    test = dataprep.test_piano_roll(training_data[0], 15, fs=5)
    # dataprep.piano_roll_to_mid_file(training_data[0], 'test1.mid', fs=5)

    # dataprep.visualize_piano_roll(training_data[0], fs=5)

    data = training_data[:3]
    generalist = Generalist([128, 43, 43, 43, 128], data, len(data), 'generalist.h5')
    generalist.train_network(10)

    path = 'generalist.h5'
    generalist.save_network(path)
    generalist_result = generalist.gen_music(test.T, fs=5)

    dataprep.visualize_piano_roll(generalist_result, fs=5)
    dataprep.piano_roll_to_mid_file(generalist_result, 'gen_res1.mid', fs=5)








if __name__ == '__main__':
    main()