import datapreparation as dataprep
import os
path = 'datasets/training/'
for file in os.listdir(path):
    if file.endswith('.mid'):
        dataprep.midfile_to_piano_roll(path + file, fs=10)

