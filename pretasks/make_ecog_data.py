#Imports signal transform functions from "Self-supervised ECG Representation Learning for Emotion Recognition" repo

import numpy as np
import math, pdb, glob, os, sys, natsort, h5py, mne
import cv2
from sklearn.model_selection import KFold
from signal_transforms import *
from sklearn.model_selection import train_test_split
import xarray as xr

np.random.seed(seed=42)

#read in speech fif files
def get_speech_epochs(sbj, lp):
    sigs, labs = [], []
    f_load = natsort.natsorted(glob.glob(lp+'/*_epo.fif'))
    print("for sjb",sbj, "f_load is",f_load)
    eps_to_file = {}
    for i, f in enumerate(f_load):
        eps = f_load[i].split('/')[-1].split('_')[3][:3]
        eps_to_file[eps] = f_load[i]
    max_eps, min_eps = max(eps_to_file, key=eps_to_file.get), min(eps_to_file, key=eps_to_file.get)
    epochs_train = mne.read_epochs(eps_to_file[max_eps]).crop(tmin=-3, tmax=3, include_tmax=True)
    epochs_test = mne.read_epochs(eps_to_file[min_eps]).crop(tmin=-3, tmax=3, include_tmax=True)
    train_dat = epochs_train.get_data()
    test_dat = epochs_test.get_data()
    return train_dat, test_dat


def get_steve_wrist_epochs(sbj, lp, event_types, n_chans_all=64, tlim=[-1,1]):
    ep_data_in = xr.open_dataset(lp+sbj+'_ecog_data.nc')
    ep_times = np.asarray(ep_data_in.time)
    time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]

    days_all_in = np.asarray(ep_data_in.events)
    days_train = np.unique(days_all_in)[:-1]
    day_test_curr = np.unique(days_all_in)[-1]
    days_test_inds = np.nonzero(days_all_in==day_test_curr)[0]
    #Compute indices of days_train in xarray dataset
    days_train_inds = []
    for day_tmp in list(days_train):
        days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])
    #Extract data
    dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_all),
                                time=time_inds)].to_array().values.squeeze()
    dat_test = ep_data_in[dict(events=days_test_inds,channels=slice(0,n_chans_all),
                                       time=time_inds)].to_array().values.squeeze()
    #if I only want one event type
    if len(event_types) == 1:
        labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                       time=0)].to_array().values.squeeze()
        labels_to_keep = np.where(labels_train == event_types[0])
        dat_train = dat_train[labels_to_keep]
    return dat_train, dat_test


#
def signal_transform(dat_train, dat_test):
    #Apply signal transformations
    transform_task = [0, 1, 2, 3, 4] #, 5, 6
    noise_amount, scaling_factor, permutation_pieces = 10, 15, 9  #
    noised_train       = add_noise(dat_train, noise_amount = noise_amount)
    scaled_train       = scaled(dat_train, factor = scaling_factor)
    negated_train      = negate(dat_train)
    flipped_train      = hor_filp(dat_train)
    #repeat for test data
    noised_test       = add_noise(dat_test, noise_amount = noise_amount)
    scaled_test       = scaled(dat_test, factor = scaling_factor)
    negated_test      = negate(dat_test)
    flipped_test      = hor_filp(dat_test)

    #Build data arrays
    train_sigs = np.vstack (( dat_train, noised_train, scaled_train, negated_train, flipped_train))
    train_labs = np.repeat(transform_task, dat_train.shape[0])
    test_sigs = np.vstack (( dat_test, noised_test, scaled_test, negated_test, flipped_test))
    test_labs = np.repeat(transform_task, dat_test.shape[0])
    return np.expand_dims(train_sigs,1), train_labs, np.expand_dims(test_sigs,1), test_labs


def rel_pos(dat_train, dat_test, T_pos, T_neg):
    rp_train, rp_train_labels = concat_positions(dat_train, T_pos, T_neg)
    rp_test, rp_test_labels = concat_positions(dat_test, T_pos, T_neg)
    return np.expand_dims(rp_train,1), rp_train_labels, np.expand_dims(rp_test,1), rp_test_labels
