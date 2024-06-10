from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D
import tensorflow as tf
from tensorflow.keras import backend as K
from DropBlock import  *

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = K.int_shape(input_feature)[-1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = K.int_shape(input_feature)[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert K.int_shape(avg_pool)[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert K.int_shape(max_pool)[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert K.int_shape(concat)[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert K.int_shape(cbam_feature)[-1] == 1
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

