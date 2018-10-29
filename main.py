import datapreparation as dataprep
import numpy as np

def main():
    dataset = 'piano_roll_fs1'

    training_data = dataprep.load_all_dataset('datasets/training/' + dataset)

    names = dataprep.load_all_dataset_names('datasets/training/' + dataset)

    test = dataprep.test_piano_roll(training_data[0], 5, fs=1)
    # dataprep.piano_roll_to_mid_file(test, 'test1.mid', fs=5)

    dataprep.visualize_piano_roll(training_data[0].T[:5].T, fs=1)

    print(training_data[0].T[1])
    print(np.where(training_data[0].T[2] == 1))
    print(names)



if __name__ == '__main__':
    main()