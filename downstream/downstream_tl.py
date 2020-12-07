import numpy as np

import os
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

# set various parameters
# data params
norm_rate = 0.25
test_day = 'last'
n_chans_all=64
tlim=[-1,1]
n_folds = 1
folds = 3

# where to grab data
wrist_lp = '/data1/users/stepeter/cnn_hilbert/ecog_data/xarray/'
pats_ids_in = ['a0f66459','c95c1e82','cb46fd46','fcb01f7a','ffb52f92','b4ac1726',
               'f3b79359','ec761078','f0bbc9a9','abdb496b','ec168864','b45e3f7b']
# model params
optimizer='adam'
loss='binary_crossentropy'
patience = 15
early_stop_monitor='val_loss'
epochs=64
sp = '/home/zsteineh/ez_ssl_results/'

acc_dict = {}

# load in each subject
for sbj in pats_ids_in:
    total_acc_lst = []
    for f in range(folds):
#         set specific params
        fold = f
        pretask_type = 'sig_tran'
        chckpt_path = sp+pretask_type+'/checkpoint_gen_tl_'+sbj+'_fold'+str(fold)+'.h5'
        model_dir = '/data1/users/gsquist/state_decoder/accuracy_outputs/'+sbj+'/class_ssl/'
        model_name = pretask_type+'_model_htnet_fold_'+str(fold)+'.h5'
        model_fname = model_dir + model_name
        
#         load in the data and model
        X,y,x_test,y_test,sbj_order_all,sbj_order_test_last = load_data(sbj, wrist_lp,
                                                              n_chans_all=n_chans_all,
                                                              test_day=test_day, tlim=tlim)
        pretask_model = tf.keras.models.load_model(model_fname)
        pretask_model.summary()
        
#         split data for test and val, and convert to tensorflow version
        nb_classes = len(np.unique(y))
        order_inds = np.arange(len(y))
        np.random.shuffle(order_inds)
        X = X[order_inds,...]
        y = y[order_inds]
        order_inds_test = np.arange(len(y_test))
        np.random.shuffle(order_inds_test)
        # X_test = X_test[order_inds_test,...]
        # y_test = y_test[order_inds_test]
        y2 = np_utils.to_categorical(y-1)
        y_test2 = np_utils.to_categorical(y_test-1)
        X2 = np.expand_dims(X,1)
        X_test2 = np.expand_dims(x_test,1)

        split_len = int(X2.shape[0]*0.2)
        last_epochs = np.zeros([n_folds,2])

        val_inds = np.arange(0,split_len)+(0*split_len)
        train_inds = np.setdiff1d(np.arange(X2.shape[0]),val_inds) #take all events not in val set

        x_train = X2[train_inds,...]
        y_train = y2[train_inds]
        x_val = X2[val_inds,...]
        y_val = y2[val_inds]
        
#         create model for transfer learning
        x = pretask_model.layers[-3].output
        # x = Flatten(name = 'flatten2')(x)
        x = Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(norm_rate))(x)
        softmax = Activation('softmax', name = 'softmax')(x)
        transfer_model = Model(inputs=pretask_model.input, outputs=softmax)

        # Set only last 3 layers to be trainable
        for l in transfer_model.layers:
            l.trainable = False
        for l in transfer_model.layers[-3:]:
            l.trainable = True #train last 3 layers
#         Can update this if desired to be other convolutions
        transfer_model.get_layer('depthwise_conv2d').trainable = True

        transfer_model.summary()
        
#         compile the model and train
        transfer_model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
        checkpointer = ModelCheckpoint(filepath=chckpt_path,verbose=1,save_best_only=True)
        early_stop = EarlyStopping(monitor=early_stop_monitor, mode='min',
                               patience=patience, verbose=0)
        h = transfer_model.fit(x_train, y_train, batch_size = 16, epochs = epochs, 
                        verbose = 2, validation_data=(x_val, y_val),
                        callbacks=[checkpointer,early_stop])
        
#         Get the accuracies. Just prints for now
        transfer_model.load_weights(chckpt_path)
        acc_lst = []
        preds = transfer_model.predict(x_train).argmax(axis = -1) 
        acc_lst.append(np.mean(preds == y_train.argmax(axis=-1)))
        preds = transfer_model.predict(x_val).argmax(axis=-1)
        acc_lst.append(np.mean(preds == y_val.argmax(axis=-1)))
        preds = transfer_model.predict(X_test2).argmax(axis = -1)
        acc_lst.append(np.mean(preds == y_test2.argmax(axis=-1)))
        
        total_acc_lst.append(acc_lst)
        
    acc_dict[sbj] = total_acc_lst
    print("subject "+sbj+" done")

print(acc_dict)
name = pretask_type+'_acc_dict'
with open(sp+'obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)


        




