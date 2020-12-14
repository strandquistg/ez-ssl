#@author: Steve Peterson, from stepeter_sandbox/DL_hilbert/rNN_project/
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import os,pdb,argparse,pickle,glob,natsort, sys, mne, datetime
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
# EEGNet-specific imports
from NN_models import cNN_state_model #eegnet,
from hilbert_DL_utils import *
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from os import path
from itertools import product
import xarray as xr
from keras.utils.vis_utils import plot_model
from mne.datasets import multimodal
from sklearn.utils import shuffle
sys.path.append('/home/gsquist/repos/ellie_sandbox/epoching_scripts')
sys.path.append('/home/gsquist/repos/ellie_sandbox/ssl_class_project/')
from load_data_utils import *
from epoching_utils import *
from signal_transforms import *
from make_ecog_data import *

#####################################
#Data Params
n_folds = 3
data_srate = 500
n_chans = 64
epoch_len = 9
T_pos, T_neg = 6, 36 #

#Model Params
dropoutType = 'Dropout'
kernLength = 64
nROIs=100
useHilbert=True
projectROIs=False
optimizer='adam'
loss='categorical_crossentropy'
#loss='binary_crossentropy'
patience = 15
early_stop_monitor='val_loss'
epochs=64
modeltype = 'htnet'    #, 'eegnet', 'eegnet_hilb', 'lstm_eegnet', 'lstm_hilb', 'rf'
datatype = "speech" #wrist
pretask = "st" #rp
train, test, X_trainval, Y_trainval, X_test, Y_test, states_all = [], [], [], [], [], [], []
chckpt_path = ""
wrist_lp = '/data1/users/stepeter/cnn_hilbert/ecog_data/xarray/'
speech_lp = '/data2/users/gsquist/speech_project/epochd_ecog_data/'
sp = '/data1/users/gsquist/state_decoder/accuracy_outputs/'

#####################################
for sbj in ['a0f66459']:
    if not os.path.exists(sp+sbj+'/class_ssl/'):
        os.makedirs(sp+sbj+'/class_ssl/')

    if datatype == "wrist":
        train, test = get_steve_wrist_epochs(sbj[:3], wrist_lp, event_types=[1, 2], n_chans_all=n_chans, tlim=[-1,1])
    elif datatype == "speech":
        train, test = get_speech_epochs(sbj, speech_lp)

    if pretask == "st":
        X_trainval, Y_trainval, X_test, Y_test = signal_transform(train, test)
        states_all = ['original_signal', 'noised_signal', 'scaled_signal', 'negated_signal', 'flipped_signal']
    elif pretask == "rp":
        X_trainval, Y_trainval, X_test, Y_test = rel_pos(train, test, T_pos, T_neg)
        states_all = ['t_pos', 't_neg']

    num_evs_per_state = np.min( np.unique(Y_trainval, return_counts=True)[1] )
    print("num events per state:",num_evs_per_state)
    ev_inds_shuffle = np.arange(num_evs_per_state)
    # Reformat for CNN fitting
    Y_trainval_cat = np_utils.to_categorical(Y_trainval)
    Y_test_cat  = np_utils.to_categorical(Y_test)

    num_evs_per_fold = num_evs_per_state//n_folds
    accs = np.zeros([n_folds,3]) # accuracy table for all NN models, Rows x Columns: Folds x Train/val/test
    last_epochs = np.zeros(n_folds)

    for i in range(n_folds):
        print("#####################################\n#####################################\nFold number",i)
        if pretask == "st":
            chckpt_path = sp+sbj+'/class_ssl/sig_tran_' + datatype + "_" + modeltype + '_fold_' + str(i)+'.h5'
        elif pretask == "rp":
            chckpt_path = sp+sbj+'/class_ssl/rel_pos_' + datatype + "_" + modeltype + '_fold_' + str(i)+'.h5'

        curr_inds = ev_inds_shuffle[(i*num_evs_per_fold):((i+1)*num_evs_per_fold)]
        val_inds = []
        for j in range(len(states_all)):
            inds_state = np.nonzero(j==Y_trainval)[0]
            inds_state = inds_state[curr_inds]
            val_inds.extend(inds_state.tolist())
        train_inds = np.setdiff1d(np.arange(X_trainval.shape[0]),np.asarray(val_inds))
        X_train = X_trainval[train_inds,...]
        X_val = X_trainval[val_inds,...]
        Y_train = Y_trainval_cat[train_inds,...]
        Y_val = Y_trainval_cat[val_inds,...]

        # Fit model
        nb_classes = len(states_all)

        model = htnet(nb_classes, Chans = X_train.shape[2], Samples = X_train.shape[-1],
                  dropoutRate = 0.5, kernLength = kernLength, F1 = 8,
                  D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
                  ROIs = 100,useHilbert=True,projectROIs=False,kernLength_sep = 16,
                  do_log=False,compute_val='power',data_srate = 500,base_split = 4)

        model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
        pdb.set_trace()
        checkpointer = ModelCheckpoint(filepath=chckpt_path,verbose=1,save_best_only=True)
        early_stop = EarlyStopping(monitor=early_stop_monitor, mode='min',
                                   patience=patience, verbose=0) #stop if val_loss doesn't improve after certain # of epochs

        fittedModel = model.fit(X_train, Y_train, batch_size = 15, epochs = epochs,
                                verbose = 2, validation_data=(X_val, Y_val),
                                callbacks=[checkpointer,early_stop])

        #Get the last epoch for training
        last_epoch = len(fittedModel.history['loss'])
        if last_epoch<epochs:
            last_epoch -= patience # revert to epoch where best model was found

        # Load model weights from best model and compute train/val/test accuracies
        model.load_weights(chckpt_path)
    #
        accs_lst = []
        #pdb.set_trace()
        preds_full = model.predict(X_train)
        preds       = model.predict(X_train).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_train.argmax(axis=-1)))

        preds_full = model.predict(X_val)
        preds       = model.predict(X_val).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_val.argmax(axis=-1)))

        preds_full = model.predict(X_test)
        preds       = model.predict(X_test).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_test_cat.argmax(axis=-1)))
        tf.keras.backend.clear_session()

        for ss in range(3):
            accs[i,ss] = accs_lst[ss]

    #Rows x Columns: Folds x Train/val/test
    print(sbj,"accs:\n",accs,"last epoch",last_epoch)
    # Save accuracies (train/val/test)
    # np.save(sp+sbj+'/pierre/acc_'+modeltype+'_cnn_state_model_'+str(num_evs_per_state)+'evts_'+str(epoch_len)+'s.npy', accs)
    # np.save(sp+sbj+'/pierre/last_training_epoch_gen_tf'+modeltype+'_cnn_state_model_'+str(num_evs_per_state)+'evts_' + str(epoch_len) + 's.npy', last_epochs)
    # print("End of training")
