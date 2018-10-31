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

    generalist = Generalist([128, 31, 31, 31, 128], training_data)
    generalist.train_network(250)
    generalist.save_network('generalist.h5')
    generalist_result = generalist.gen_music(test.T)

    dataprep.visualize_piano_roll(generalist_result, fs=5)
    dataprep.piano_roll_to_mid_file(generalist_result, 'gen_res.mid', fs=5)








if __name__ == '__main__':
    main()