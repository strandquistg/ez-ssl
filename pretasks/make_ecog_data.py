#Imports signal transform functions from "Self-supervised ECG Representation Learning for Emotion Recognition" repo

import numpy as np
import math, pdb, glob, os, sys, natsort, h5py, mne
import cv2
from sklearn.model_selection import KFold
from signal_transforms import *
from sklearn.model_selection import train_test_split

#apply signal transformations to epoched ecog data,
#return in train/test format to use in HTNet
def speech_signal_transform(sbj, lp):
    sigs, labs = [], []
    transform_task = [0, 1, 2, 3, 4] #, 5, 6
    noise_amount, scaling_factor, permutation_pieces = 10, 15, 9  #
    f_load = natsort.natsorted(glob.glob(lp+'/*_epo.fif'))
    print("in signal transform func", lp+sbj,"\nwith files",f_load)
    for i, f in enumerate(f_load):
        epochsAll = mne.read_epochs(f_load[i])#.crop(tmin=0, tmax=3, include_tmax=True)
        #labs = epochsAll.events[:,-1].copy()
        signal = epochsAll.get_data()
        noised_signal       = add_noise(signal, noise_amount = noise_amount) #round(np.random.uniform(0.005,0.05),2)) # 0.005 - 0.05
        scaled_signal       = scaled(signal, factor = scaling_factor) #round(np.random.uniform(0.2,2),2)) # 0.2 - 2
        negated_signal      = negate(signal)
        flipped_signal      = hor_filp(signal)
        #permuted_signal     = permute(signal, pieces = permutation_pieces) # 2-20
        # time_warped_signal = time_warp(signal, sampling_freq, pieces = time_warping_pieces, stretch_factor = time_warping_stretch_factor, squeeze_factor = time_warping_squeeze_factor)
        total_sigs = np.vstack (( signal, noised_signal, scaled_signal, negated_signal, flipped_signal))
        total_labs = np.repeat(transform_task, signal.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(total_sigs, total_labs, stratify=total_labs, test_size=0.33)
    return np.expand_dims(X_train,1), y_train, np.expand_dims(X_test,1), y_test
