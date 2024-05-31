# coding=utf-8
from unittest import case
import h5py, os
import numpy
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf
import scipy.io as scio
import datetime, copy
from pypower.loadcase import loadcase
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
from pypower.idx_brch import PF, QF, BR_STATUS, RATE_A, RATE_B, RATE_C, BR_R, BR_B, BR_X, T_BUS, F_BUS
from scipy.io import loadmat
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from pypower.makePTDF import makePTDF
import csv


def data_generator(data, batch_size=15):
    idx = np.random.permutation(len(data))
    for k in range(int(np.ceil(len(data) / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        idx_ = idx[from_idx:to_idx]
        inputs_ = np.zeros((idx_.shape[0], len(data[0][1]['pd'])))
        outputs = np.zeros((idx_.shape[0], len(data[0][1]['pg'])))
        for i in range(idx_.shape[0]):
            inputs_[i, :] = np.array(data[idx_[i]][1]['pd'])[:, 0]
            outputs[i, :] = np.array(data[idx_[i]][1]['pg'])[:, 0]

        yield tf.Variable(inputs_, name='x', dtype=tf.float32), \
              tf.constant(outputs, dtype=tf.float32), \


class Last_Layer(keras.layers.Layer):
    def __init__(self, W, B,  **kwargs):
        super(Last_Layer, self).__init__(**kwargs)
        self.W = tf.constant(W, dtype='float32')
        self.B = tf.constant(B, dtype='float32')

    def call(self, seta):
        fs = self.W*seta + self.B
        return fs


def my_model():
    node_num = 30
    input_PD = keras.Input(shape=(node_num,), name='input_PD')
    characteristic = keras.layers.Dense(units=200, activation='relu')(input_PD)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    PG_out = keras.layers.Dense(units=6, name='output_PG')(characteristic)
    # outs = Last_Layer(W, B, name='outputs')(PG_out)

    model = keras.Model(inputs=input_PD, outputs=PG_out)
    model.summary()
    # model.compile()

    return model


def train_per_step(fs):
    temp = tf.Variable(fs[0], name='x', dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape1:
        predicts = model(temp)
        # pg的有监督loss
        loss_value = loss(fs[1], predicts)
    # 根据损失求梯度
    gradients1 = tape1.gradient(loss_value, model.trainable_variables)
    # 把梯度和变量进行绑定
    grads_and_vars = zip(gradients1, model.trainable_variables)
    # 进行梯度更新
    my_opt.apply_gradients(grads_and_vars)

    return loss_value


if __name__ == '__main__':
    import tensorflow as tf
    from pypower.case30 import case30
    import os

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    batch_size = 50
    si_coff = 0.001
    dir_case = './models/M1/'
    traning_data_dir = ''
    testing_data_dir = ''
    testing_data_dir2 = ''

    # load data
    with open(traning_data_dir) as json_file:
        trainData = json.load(json_file)
        trainData = list(trainData.items())
    with open(testing_data_dir) as json_file:
        testData = json.load(json_file)
        testData = list(testData.items())

    with open(testing_data_dir2) as json_file:
        testData2 = json.load(json_file)
        testData2 = list(testData2.items())

    print("training data: {}，testing data1: {}，testing data2: {}".format(len(trainData), len(testData), len(testData2)))

    # build neural network
    model = my_model()

    # configuration
    loss = tf.keras.losses.MeanAbsoluteError()
    my_opt = keras.optimizers.Adam(learning_rate=0.001, decay=2e-6)

    saver = tf.train.Checkpoint(optimizer=my_opt, model=model)
    model.compile(optimizer=my_opt, loss=loss)

    if not os.path.exists(dir_case):
        os.makedirs(dir_case)
    f = open(dir_case + 'losses.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["epoch", "total_loss", "test1_loss", "test2_loss"])   # 构建列表头
    losses = []

    # training
    for epoch in range(1000):
        epoch_loss_avg = []
        for fs in data_generator(trainData, batch_size=batch_size):
            loss_value = train_per_step(fs)
            epoch_loss_avg.append(loss_value.numpy())

        test_loss = []
        for fs in data_generator(testData, batch_size=batch_size):
            predicts = model(fs[0])
            loss_value1 = loss(fs[1], predicts)
            test_loss.append(loss_value1)

        test2_loss = []
        for fs in data_generator(testData2, batch_size=batch_size):
            predicts = model(fs[0])
            loss_value1 = loss(fs[1], predicts)
            test2_loss.append(loss_value1)

        csv_writer.writerow([epoch, np.mean(epoch_loss_avg), np.mean(test_loss), np.mean(test2_loss)])
        print("Epoch {:03d}: Loss: {:.6f}, test1_loss:{:.6f}, test2_loss:{:.6f} ".format(epoch,
                                                                                         np.mean(epoch_loss_avg),
                                                                                         np.mean(test_loss),
                                                                                         np.mean(test2_loss)))

    f.close()

    # save model
    model.save('./{}/Model.h5'.format(dir_case))








