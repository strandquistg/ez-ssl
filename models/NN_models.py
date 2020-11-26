#@author: Steve Peterson, from stepeter_sandbox/DL_hilbert/rNN_project/
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D,LSTM,Permute
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten, Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import sys
sys.path.append('/home/stepeter/AJILE/cnn_hilbert/cnn_hilbert_workspace/')
from hilbert_DL_utils import *
from tensorflow import squeeze as tf_squeeze


def cNN_state_model(nb_classes, Chans = 64, Samples = 128,
                    dropoutRate = 0.5, kernLength = 64, D=2, F1 = 8,
                    norm_rate = 0.25, dropoutType = 'Dropout',
                    ROIs = 100,useHilbert=False,projectROIs=False,
                    do_log=False,compute_val='power',ecog_srate=500):
    """
    CNN state model
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1   = Input(shape = (1, Chans, Samples))

    if projectROIs:
        input2   = Input(shape = (1, ROIs, Chans))

    ##################################################################
    X1 = Conv2D(F1, (1, kernLength), padding = 'same',
               input_shape = (1, Chans, Samples),
               use_bias = False)(input1)

    X1 = Lambda(apply_hilbert_tf, arguments={'do_log':do_log,'compute_val':compute_val,\
                                            'ecog_srate':ecog_srate})(X1) #Hilbert transform

    X1 = AveragePooling2D((1, X1.shape[-1]))(X1) # average across all time points #(None, 8, 64, 1)

    # Compute total power
    X2 = Lambda(apply_hilbert_tf, arguments={'do_log':do_log,'compute_val':compute_val,\
                                             'ecog_srate':ecog_srate})(input1) #Hilbert transform

    X2 = AveragePooling2D((1, X2.shape[-1]))(X2) # average across all time points
    X2 = Lambda(lambda x: tf.tile(x,tf.constant([1,F1,1,1], dtype=tf.int32)))(X2)


    # Divide filtered by total power at every electrode to get relative value
    X = Lambda(lambda inputs: tf.math.truediv(inputs[0],inputs[1]))([X1, X2])

    if projectROIs:
        X = Lambda(proj_to_roi)([X,input2]) #project to ROIs
    X = BatchNormalization(axis = 1)(X)
    if projectROIs:
        X = DepthwiseConv2D((ROIs, 1), use_bias = False,
                            depth_multiplier = D,
                            depthwise_constraint = max_norm(1.))(X)
    else:
        X = DepthwiseConv2D((Chans, 1), use_bias = False,
                            depth_multiplier = D,
                            depthwise_constraint = max_norm(1.))(X)

    X = BatchNormalization(axis = 1)(X)
    X = Activation('elu')(X)
    X = dropoutType(dropoutRate)(X)
    X = Flatten(name = 'flatten')(X)
    X = Dense(nb_classes, kernel_constraint = max_norm(norm_rate))(X)
    softmax = Activation('softmax', name = 'softmax')(X)

    if projectROIs:
        print("projectROIs")
        return Model(inputs=[input1,input2], outputs=softmax)
    else:
        print("Not projectROIs")
        return Model(inputs=input1, outputs=softmax)
