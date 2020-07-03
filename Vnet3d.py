from GroupNorm3D import GroupNormalization

import warnings
from keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Activation, Add, Concatenate, Lambda, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


def dice_loss_v1(gt, pred):
    gt = tf.cast(gt, tf.float32)
    sum_gt = tf.reduce_sum(gt, axis=(1, 2, 3, 4))
    sum_pred = tf.reduce_sum(pred, axis=(1, 2, 3, 4))

    sum_dot = tf.reduce_sum(gt * pred, axis=(1, 2, 3, 4))
    epsilon = 1e-6
    dice = (2. * sum_dot + epsilon) / (sum_gt + sum_pred + epsilon)
    dice_loss = 1 - tf.reduce_mean(dice, name='dice_loss')
    return dice_loss

def dice_loss_v2(gt, pred, numClass=3):
    
    losses = []
    for class_id in range(numClass):
        gt_id = gt[:, class_id, :, :, :]
        pred_id = pred[:, class_id, :, :, :]
        
        gt_id = tf.cast(gt_id, tf.float32)
        
        sum_gt = tf.reduce_sum(gt_id)
        sum_pred = tf.reduce_sum(pred_id)
        
        sum_dot = tf.reduce_sum(gt_id * pred_id)
        epsilon = 1e-6
        dice = (2. * sum_dot + epsilon) / (sum_gt + sum_pred + epsilon)
        dice_loss = 1.0 - tf.reduce_mean(dice)
        
        if class_id == 0:
            losses.append(dice_loss)
        else:
            losses.append(dice_loss)
    
    return tf.reduce_mean(tf.stack(losses))

def dice_loss_v3(gt, pred):
    gt = tf.cast(gt, tf.float32)
    sum_gt = tf.reduce_sum(gt, axis=(2, 3, 4))
    sum_pred = tf.reduce_sum(pred, axis=(2, 3, 4))

    sum_dot = tf.reduce_sum(gt * pred, axis=(2, 3, 4))

    epsilon = 1e-6
    dice = (2.0 * sum_dot + epsilon) / (sum_gt + sum_pred + epsilon)
    dice_loss = tf.reduce_mean(tf.reduce_mean(1.0 - dice, axis=-1), name='dice_loss')
    return dice_loss

def hybrid_loss(gt, pred):
    return dice_loss(gt, pred)

def vnet(num_input_channel, base_size, numofclasses):

    # Layer 1
    if data_format == 'channels_first':
        inputs = Input([num_input_channel,] + [base_size, base_size, base_size])
    else:
        inputs = Input([base_size, base_size, base_size] + [num_input_channel,])
    
    conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=16, axis=1)(conv1)
    conv1 = PReLU()(conv1)

    identity1 = Conv3D(16, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(inputs)
    identity1 = GroupNormalization(groups=16, axis=1)(identity1)
    identity1 = PReLU()(identity1)

    conv1 = Add()([conv1, identity1])

    down1 = Conv3D(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv1)
    down1 = PReLU()(down1)

    conv2 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down1)
    conv2 = GroupNormalization(groups=16, axis=1)(conv2)
    conv2 = PReLU()(conv2)

    conv2 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=16, axis=1)(conv2)
    conv2 = PReLU()(conv2)

    identity2 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down1)
    identity2 = GroupNormalization(groups=16, axis=1)(identity2)
    identity2 = PReLU()(identity2)

    conv2 = Add()([conv2, identity2])

    down2 = Conv3D(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv2)
    down2 = PReLU()(down2)

    conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down2)
    conv3 = GroupNormalization(groups=16, axis=1)(conv3)
    conv3 = PReLU()(conv3)

    conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=16, axis=1)(conv3)
    conv3 = PReLU()(conv3)

    conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=16, axis=1)(conv3)
    conv3 = PReLU()(conv3)
    
    identity3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down2)
    identity3 = GroupNormalization(groups=16, axis=1)(identity3)
    identity3 = PReLU()(identity3)

    conv3 = Add()([conv3, identity3])

    down3 = Conv3D(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    down3 = GroupNormalization(groups=16, axis=1)(down3)
    down3 = PReLU()(down3)

    conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down3)
    conv4 = GroupNormalization(groups=16, axis=1)(conv4)
    conv4 = PReLU()(conv4)

    conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    conv4 = GroupNormalization(groups=16, axis=1)(conv4)
    conv4 = PReLU()(conv4)

    conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    conv4 = GroupNormalization(groups=16, axis=1)(conv4)
    conv4 = PReLU()(conv4)
    
    identity4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down3)
    identity4 = GroupNormalization(groups=16, axis=1)(identity4)
    identity4 = PReLU()(identity4)

    conv4 = Add()([conv4, identity4])

    down4 = Conv3D(256, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    down4 = PReLU()(down4)

    conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down4)
    conv5 = GroupNormalization(groups=16, axis=1)(conv5)
    conv5 = PReLU()(conv5)

    conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=16, axis=1)(conv5)
    conv5 = PReLU()(conv5)

    conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=16, axis=1)(conv5)
    conv5 = PReLU()(conv5)
    
    identity5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down4)
    identity5 = GroupNormalization(groups=16, axis=1)(identity5)
    identity5 = PReLU()(identity5)

    conv5 = Add()([conv5, identity5])

    up1 = Conv3DTranspose(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    concat1 = Concatenate(axis=1)([up1, conv4])

    conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat1)
    conv6 = GroupNormalization(groups=16, axis=1)(conv6)
    conv6 = PReLU()(conv6)

    conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=16, axis=1)(conv6)
    conv6 = PReLU()(conv6)

    conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=16, axis=1)(conv6)
    conv6 = PReLU()(conv6)
    
    identity6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up1)
    identity6 = GroupNormalization(groups=16, axis=1)(identity6)
    identity6 = PReLU()(identity6)
    
    conv6 = Add()([conv6, identity6])

    up2 = Conv3DTranspose(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    concat2 = Concatenate(axis=1)([up2, conv3])

    conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat2)
    conv7 = GroupNormalization(groups=16, axis=1)(conv7)
    conv7 = PReLU()(conv7)

    conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=16, axis=1)(conv7)
    conv7 = PReLU()(conv7)

    conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=16, axis=1)(conv7)
    conv7 = PReLU()(conv7)
    
    identity7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up2)
    identity7 = GroupNormalization(groups=16, axis=1)(identity7)
    identity7 = PReLU()(identity7)
    
    conv7 = Add()([conv7, identity7])

    up3 = Conv3DTranspose(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    concat3 = Concatenate(axis=1)([up3, conv2])

    conv8 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat3)
    conv8 = GroupNormalization(groups=16, axis=1)(conv8)
    conv8 = PReLU()(conv8)

    conv8 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv8)
    conv8 = GroupNormalization(groups=16, axis=1)(conv8)
    conv8 = PReLU()(conv8)

    identity8 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up3)
    identity8 = GroupNormalization(groups=16, axis=1)(identity8)
    identity8 = PReLU()(identity8)
    
    conv8 = Add()([conv8, identity8])

    up4 = Conv3DTranspose(16, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv8)
    concat4 = Concatenate(axis=1)([up4, conv1])

    conv9 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat4)
    conv9 = GroupNormalization(groups=16, axis=1)(conv9)
    conv9 = PReLU()(conv9)
    
    identity9 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up4)
    identity9 = GroupNormalization(groups=16, axis=1)(identity9)
    identity9 = PReLU()(identity9)
    
    conv9 = Add()([conv9, identity9])

    logits = Conv3D(numofclasses, kernel_size=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv9)
    output1 = Lambda(lambda x: K.softmax(x, axis=1))(logits)

    gt = Input((numofclasses, base_size, base_size, base_size))

    seg_loss = Lambda(lambda x: hybrid_loss(*x, numClass=numofclasses), name="seg_loss")([gt, output1])
    model = Model(inputs=[inputs, gt], outputs=[output1, seg_loss])
    model.add_loss(lambda_seg * seg_loss)
    model.compile(optimizer=Adam(lr=0.0001), loss=[None] * len(model.outputs))
    
    metrics_names = ["seg_loss"]
    loss_weights = {
        "seg_loss": lambda_seg,
    }
    
    for name in metrics_names:
        layer = model.get_layer(name)
        loss = (layer.output * loss_weights.get(name, 1.))
        model.metrics_tensors.append(loss)

    return model

if __name__ == '__main__':
    model = vnet(1, 64, 3)
    model.summary()














