import sys, pdb, random
import numpy as np

#adapted from https://github.com/mlberkeley/eeg-ssl/tree/aab50b6c6858d3596cb22e31786783dc0f620e36
#Take an epoch, and concat it with examples of close-by/far-away windows
#This concated chunk will be a "single" input into HTNet now
def concat_positions(epochs, T_pos, T_neg):
    total_samples = epochs.shape[0] * 6
    RP_dataset = np.empty((total_samples, epochs.shape[1], epochs.shape[2]*2))
    RP_labels = np.empty((total_samples, 1))
    counter = 0

    for idx, sample1 in enumerate(epochs):
        for _ in range(3): # Loop for T_pos
            sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            while sample2_index == idx: # should not be the same
                sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            sample2 = epochs[sample2_index]

            y = 0
            RP_sample = np.concatenate([sample1, sample2], axis=-1)
            RP_dataset[counter] = RP_sample
            RP_labels[counter] = y
            counter += 1

        for _ in range(3): # Loop for T_neg
            #if we're on the early few epochs I think?
            if idx-T_neg <= 0: # T_neg if (corners)
                sample2_index = np.random.randint(idx+T_neg, epochs.shape[0])
            #nearing the last epochs?
            elif idx+T_neg >= epochs.shape[0]: # take care of low == high
                sample2_index = np.random.randint(0, idx-T_neg)
            else:
                #I think this randomly is supposed to select a negative window from either before or after the current epoch
                sample2_index_1 = np.random.randint(idx+T_neg, epochs.shape[0])
                sample2_index_2 = np.random.randint(0, idx-T_neg)
                sample2_index = list([sample2_index_1, sample2_index_2])[int(random.uniform(0,1))] #only selects 0, is this a bug?...
            sample2 = epochs[sample2_index]

            y = 1

            RP_sample = np.concatenate([sample1, sample2], axis=-1)
            RP_dataset[counter] = RP_sample
            RP_labels[counter] = y
            counter += 1

    return RP_dataset, RP_labels
