#Imports signal transform functions from "Self-supervised ECG Representation Learning for Emotion Recognition" repo

import numpy as np
import math, pdb, glob, os, sys, natsort, h5py, mne
import cv2
from sklearn.model_selection import KFold
from signal_transforms import *
from sklearn.model_selection import train_test_split
import xarray as xr

#apply signal transformations to epoched ecog data,
#return in train/test format to use in HTNet
def speech_signal_transform(sbj, lp):
    sigs, labs = [], []
    transform_task = [0, 1, 2, 3, 4] #, 5, 6
    noise_amount, scaling_factor, permutation_pieces = 20, 15, 9  #
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


def get_steve_wrist_epochs(sbj, lp, event_types, n_chans_all=64, tlim=[-1,1]):
    ep_data_in = xr.open_dataset(lp+sbj+'_ecog_data.nc')
    ep_times = np.asarray(ep_data_in.time)
    time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]
    days_all_in = np.asarray(ep_data_in.events)
    days_train = np.unique(days_all_in)[:-1]
    #Compute indices of days_train in xarray dataset
    days_train_inds = []
    for day_tmp in list(days_train):
        days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])
    #Extract data
    dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_all),
                                time=time_inds)].to_array().values.squeeze()
    #if I only want one event type
    if len(event_types) == 1:
        labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                       time=0)].to_array().values.squeeze()
        labels_to_keep = np.where(labels_train == event_types[0])
        dat_train = dat_train[labels_to_keep]
    return dat_train


#
def wrist_signal_transform(sbj, lp, event_types, n_chans_all=64, tlim=[-1,1]):
    #Extract data
    dat_train = get_steve_wrist_epochs(sbj, lp, event_types, n_chans_all=64, tlim=[-1,1])

    #Apply signal transformations
    transform_task = [0, 1, 2, 3, 4] #, 5, 6
    noise_amount, scaling_factor, permutation_pieces = 10, 15, 9  #
    noised_signal       = add_noise(dat_train, noise_amount = noise_amount) #round(np.random.uniform(0.005,0.05),2)) # 0.005 - 0.05
    scaled_signal       = scaled(dat_train, factor = scaling_factor) #round(np.random.uniform(0.2,2),2)) # 0.2 - 2
    negated_signal      = negate(dat_train)
    flipped_signal      = hor_filp(dat_train)
    #permuted_signal     = permute(signal, pieces = permutation_pieces) # 2-20
    # time_warped_signal = time_warp(dat_train, 500, pieces = time_warping_pieces, stretch_factor = time_warping_stretch_factor, squeeze_factor = time_warping_squeeze_factor)

    #Build data arrays
    train_sigs = np.vstack (( dat_train, noised_signal, scaled_signal, negated_signal, flipped_signal))
    train_labs = np.repeat(transform_task, dat_train.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(train_sigs, train_labs, stratify=train_labs, test_size=0.33)
    return np.expand_dims(X_train,1), y_train, np.expand_dims(X_test,1), y_test


def wrist_rel_pos(sbj, lp, T_pos, T_neg, event_types, n_chans_all=64, tlim=[-1,1]):
    dat_train = get_steve_wrist_epochs(sbj, lp, event_types, n_chans_all=n_chans_all, tlim=tlim)
    rp_data, rp_labels = concat_positions(dat_train, T_pos, T_neg)
    X_train, X_test, y_train, y_test = train_test_split(rp_data, rp_labels, stratify=rp_labels, test_size=0.33)
    return np.expand_dims(X_train,1), y_train, np.expand_dims(X_test,1), y_test
