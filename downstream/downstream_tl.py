import numpy as np

import os, math, pdb, glob, os, sys, natsort, h5py, mne
#Choose GPU 0 as a default
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import sys
sys.path.append('/home/zsteineh/cnn_hilbert/cnn_hilbert_workspace')
import hilbert_DL_utils
from hilbert_DL_utils import load_data
import pickle

def load_speech_labled(sbj, speech_lp):
    sigs, labs = [], []
    f_load = natsort.natsorted(glob.glob(speech_lp+sbj+'/speak_silent/'+'*_epo.fif'))
    print("for sjb",sbj, "f_load is",f_load)
    eps_to_file = {}
    for i, f in enumerate(f_load):
        eps = f_load[i].split('/')[-1].split('_')[3][:3]
        eps_to_file[eps] = f_load[i]
    max_eps, min_eps = max(eps_to_file, key=eps_to_file.get), min(eps_to_file, key=eps_to_file.get)
    epochs_train = mne.read_epochs(eps_to_file[max_eps]).crop(tmin=-3, tmax=3, include_tmax=True)
    epochs_test = mne.read_epochs(eps_to_file[min_eps]).crop(tmin=-3, tmax=3, include_tmax=True)
    X = epochs_train.get_data()
    x_test = epochs_test.get_data()

    y = epochs_train.events[:,-1]
    y_test = epochs_test.events[:,-1]
    
    return X, y, x_test, y_test

def load_and_split_data(sbj, lp, n_chans_all, test_day, tlim, n_folds, datatype):
    """
    Loads in the data for the specific subject, 
    and returns the data split into train, val and test in a format that 
    the tensorflow model can accept
    """
    # load in the data for the subject
    if datatype == "speech": 
        X,y,x_test,y_test = load_speech_labled(sbj, lp)
    else:
        X,y,x_test,y_test,sbj_order_all,sbj_order_test_last = load_data(sbj, lp,
                                                              n_chans_all=n_chans_all,
                                                              test_day=test_day, tlim=tlim)
    #split data for test and val, and convert to tensorflow version
    nb_classes = len(np.unique(y))
    order_inds = np.arange(len(y))
    np.random.shuffle(order_inds)
    X = X[order_inds,...]
    y = y[order_inds]
    order_inds_test = np.arange(len(y_test))
    np.random.shuffle(order_inds_test)
    x_test = x_test[order_inds_test,...]
    y_test = y_test[order_inds_test]
    y2 = np_utils.to_categorical(y-1)
    y_test2 = np_utils.to_categorical(y_test-1)
    X2 = np.expand_dims(X,1)
    X_test2 = np.expand_dims(x_test,1)

    split_len = int(X2.shape[0]*0.2)
    last_epochs = np.zeros([n_folds,2])

    val_inds = np.arange(0,split_len)+(0*split_len)
    #take all events not in val set
    train_inds = np.setdiff1d(np.arange(X2.shape[0]),val_inds) 

    x_train = X2[train_inds,...]
    y_train = y2[train_inds]
    x_val = X2[val_inds,...]
    y_val = y2[val_inds]
    
    return x_train, y_train, x_val, y_val, X_test2, y_test2, nb_classes

def create_tl_model(pretask_type, model_dir, model_fname, fold, nb_classes, norm_rate, pretask_model, datatype):
    """
    creates model for transfer learning by loading in the weights 
    from the pretask model
    """
    # if the model is pretask, we need to do something a little 
    # janky to make the dimensions work
    if pretask_type == 'rel_pos':
        # find the associated sig_tran pretask, as the dimensions here match better
        sig_tran_model_fname = model_dir + 'sig_tran_' + datatype + '_htnet_fold_'+str(fold)+'.h5'
        sig_tran_pretask_model = tf.keras.models.load_model(sig_tran_model_fname)
        x = sig_tran_pretask_model.layers[-4].output
        x = Flatten(name = 'flatten2')(x)
        x = Dense(nb_classes, name = 'dense2', kernel_constraint = max_norm(norm_rate))(x)
        softmax = Activation('softmax', name = 'softmax2')(x)

        transfer_model = Model(inputs=sig_tran_pretask_model.input, outputs=softmax)
        # here we actually load in the rel_pos pretask weights
        transfer_model.load_weights(model_fname, by_name=True)
    # otherwise we just load the model in as expected
    else:
        x = pretask_model.layers[-3].output
        # x = Flatten(name = 'flatten2')(x)
        x = Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(norm_rate))(x)
        softmax = Activation('softmax', name = 'softmax')(x)

        transfer_model = Model(inputs=pretask_model.input, outputs=softmax)

    # First set last 3 layers to be trainable since we replaced these
    for l in transfer_model.layers:
        l.trainable = False
    for l in transfer_model.layers[-3:]:
        l.trainable = True #train last 3 layers
    # Can update this if desired to be other convolutions
    # Also sets this to trainable for better performance, 
    # as per what we learned in the original HTNet paper
    transfer_model.get_layer('depthwise_conv2d').trainable = True

    transfer_model.summary()
    
    return transfer_model

def train_model(model, loss, optimizer, chckpt_path, early_stop_monitor, \
                patience, x_train, y_train, epochs, x_val, y_val):
    """
    trains the given model on the provided data with the specified loss and optimizer
    """
    model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
    checkpointer = ModelCheckpoint(filepath=chckpt_path,verbose=1,save_best_only=True)
    early_stop = EarlyStopping(monitor=early_stop_monitor, mode='min',
                           patience=patience, verbose=0)
    h = model.fit(x_train, y_train, batch_size = 16, epochs = epochs, 
                    verbose = 2, validation_data=(x_val, y_val),
                    callbacks=[checkpointer,early_stop])
    
def calc_accs(model, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Calculate train, val and test accuracies for the given model
    """
    acc_lst = []
    preds = model.predict(x_train).argmax(axis = -1) 
    acc_lst.append(np.mean(preds == y_train.argmax(axis=-1)))
    preds = model.predict(x_val).argmax(axis=-1)
    acc_lst.append(np.mean(preds == y_val.argmax(axis=-1)))
    preds = model.predict(x_test).argmax(axis = -1)
    acc_lst.append(np.mean(preds == y_test.argmax(axis=-1)))
    
    return acc_lst

def pickle_save_accs(acc_dict, pretask_type, model_type, datatype, sp):
    print(model_type)
    print(acc_dict)
    name = pretask_type+datatype+model_type+'_acc_dict'
    with open(sp+'obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(acc_dict, f)

def main():
    # set various parameters
    # data params
    norm_rate = 0.25
    test_day = 'last'
    n_chans_all=64
    tlim=[-1,1]
    n_folds = 1
    folds = 3

    # where to grab data and who
    wrist_lp = '/data1/users/stepeter/cnn_hilbert/ecog_data/xarray/'
    speech_lp = '/data2/users/gsquist/speech_project/epochd_ecog_data/'
    
    datatype = "speech" #model
    if datatype == "speech":
        pats_ids_in = ['a0f66459','cb46fd46','b45e3f7b']
        lp = speech_lp
    else:
        pats_ids_in = ['a0f66459','c95c1e82','cb46fd46','fcb01f7a','ffb52f92','b4ac1726',
                   'f3b79359','ec761078','f0bbc9a9','abdb496b','ec168864','b45e3f7b']
        lp = wrist_lp
    
    # model params
    optimizer='adam'
    loss='binary_crossentropy'
    patience = 15
    early_stop_monitor='val_loss'
    epochs=64
    sp = '/home/zsteineh/ez_ssl_results/'
    model_types = ['pretask', 'tl', 'ts']

    # dictionaries to save accuracies
    pretask_acc_dict = {}
    tl_acc_dict = {}
    ts_acc_dict = {}

    # load in each subject
    for sbj in pats_ids_in:
        pretask_total_acc_lst = []
        tl_total_acc_lst = []
        ts_total_acc_lst = []
        for fold in range(folds):
    #         set specific params
            pretask_type = 'sig_tran' #rel_pos or sig_tran
            tl_chckpt_path = sp+pretask_type+'/checkpoint_gen_tl_'+datatype+'_'+sbj+'_fold'+str(fold)+'.h5'
            ts_chckpt_path = sp+pretask_type+'/checkpoint_gen_ts_'+datatype+'_'+sbj+'_fold'+str(fold)+'.h5'
            model_dir = '/data1/users/gsquist/state_decoder/accuracy_outputs/'+sbj+'/class_ssl/'
            model_name = pretask_type+'_'+datatype+'_htnet_fold_'+str(fold)+'.h5'
            model_fname = model_dir + model_name

            pretask_model = tf.keras.models.load_model(model_fname)
            pretask_model.summary()

            # load in the data
            x_train, y_train, x_val, y_val, x_test, y_test, nb_classes = load_and_split_data(sbj, lp, \
                                                                                 n_chans_all, test_day, tlim, n_folds, datatype)

            # create the TL model
            transfer_model = create_tl_model(pretask_type, model_dir, model_fname, fold, nb_classes, \
                                             norm_rate, pretask_model, datatype)

            # get accuracies before finetuning
            pretask_total_acc_lst.append(calc_accs(transfer_model, x_train, y_train, x_val, y_val, x_test, y_test))
            
            # compile the model and finetune
            train_model(transfer_model, loss, optimizer, tl_chckpt_path, early_stop_monitor, \
                        patience, x_train, y_train, epochs, x_val, y_val)

            # Get the tl accuracies
            transfer_model.load_weights(tl_chckpt_path)
            tl_total_acc_lst.append(calc_accs(transfer_model, x_train, y_train, x_val, y_val, x_test, y_test))
            
            # Run the traditional supervised model
            ts_model = tf.keras.models.clone_model(transfer_model)
            train_model(ts_model, loss, optimizer, ts_chckpt_path, early_stop_monitor, \
                        patience, x_train, y_train, epochs, x_val, y_val)
            
            ts_model.load_weights(ts_chckpt_path)
            ts_total_acc_lst.append(calc_accs(ts_model, x_train, y_train, x_val, y_val, x_test, y_test))
            

        pretask_acc_dict[sbj] = pretask_total_acc_lst
        tl_acc_dict[sbj] = tl_total_acc_lst
        ts_acc_dict[sbj] = ts_total_acc_lst
        print("subject "+sbj+" done")

    pickle_save_accs(pretask_acc_dict, pretask_type, model_types[0], datatype, sp)
    pickle_save_accs(tl_acc_dict, pretask_type, model_types[1], datatype, sp)
    pickle_save_accs(ts_acc_dict, pretask_type, model_types[2], datatype, sp)


if __name__ == "__main__":
    main()





