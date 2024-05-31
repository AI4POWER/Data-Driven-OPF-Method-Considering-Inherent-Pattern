
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
        jacobi = np.zeros((idx_.shape[0], len(data[0][1]['pg']), len(data[0][1]['pd'])))
        k = np.ones((idx_.shape[0], 1, 1))
        active_idx = np.zeros((idx_.shape[0], 1))
        for i in range(idx_.shape[0]):
            inputs_[i, :] = np.array(data[idx_[i]][1]['pd'])[:, 0]
            outputs[i, :] = np.array(data[idx_[i]][1]['pg'])[:, 0]
            if 'branch_high_7' in data[idx_[i]][1]['info'] or 'branch_low_7' in data[idx_[i]][1]['info']:
                active_idx[i, :] = 1
            if np.sum(data[idx_[i]][1]['coff'])!= 0 and np.max(data[idx_[i]][1]['coff'])<10.0 :
                jacobi[i, ::] = np.array(data[idx_[i]][1]['coff'])
            else:
                k[i] = 0

        yield tf.Variable(inputs_, name='x', dtype=tf.float32), \
              tf.constant(outputs, dtype=tf.float32), \
              tf.constant(jacobi, dtype=tf.float32), \
              tf.constant(k, dtype=tf.float32),\
              tf.constant(active_idx, dtype=tf.float32)


class Last_Layer(keras.layers.Layer):
    def __init__(self, W, B,  **kwargs):
        super(Last_Layer, self).__init__(**kwargs)
        self.W = tf.constant(W, dtype='float32')
        self.B = tf.constant(B, dtype='float32')

    def call(self, seta):
        fs = self.W*seta + self.B
        return fs


def my_model():
    node_num = 57
    input_PD = keras.Input(shape=(node_num,), name='input_PD')
    characteristic = keras.layers.Dense(units=200, activation='relu')(input_PD)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    PG_out = keras.layers.Dense(units=7, name='output_PG')(characteristic)
    # outs = Last_Layer(W, B, name='outputs')(PG_out)

    model = keras.Model(inputs=input_PD, outputs=PG_out)
    model.summary()
    # model.compile()

    return model


def physics_per_step(fs):
    temp = tf.Variable(fs[0], name='x', dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            predicts = tf.multiply(model(temp), s)
        # 计算神经网络的雅可比矩阵
        J_model = tape1.gradient(predicts, temp)
        loss_value2 = loss(tf.multiply(J_model, fs[-1]), tf.matmul(fs[-1], s_))

        # 根据损失求梯度
    gradients2 = tape2.gradient(loss_value2, model.trainable_variables)
    vars_ = model.trainable_variables
    # 二次梯度最后一层的b的梯度为None，需要剔除
    gradients2[-1] = tf.constant(np.zeros(vars_[-1].shape), dtype=tf.float32)

    return gradients2, loss_value2


def train_per_step(fs, g):
    gradients5, loss_value5 = physics_per_step(fs)

    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            predicts = model(fs[0])
            # pg的有监督loss
            loss_value1 = loss(fs[1], predicts)
        # 计算神经网络的雅可比矩阵
        J_model = tape1.batch_jacobian(predicts, fs[0])
        loss_value2 = tf.reduce_mean(tf.multiply(tf.square(J_model - fs[2]), fs[3]))

        J_model3 = tape1.gradient(predicts, fs[0])
        loss_value3 = loss(J_model3, 1.0)

    # 根据损失求梯度
    gradients1 = tape2.gradient(loss_value1, model.trainable_variables)
    gradients2 = tape2.gradient(loss_value2, model.trainable_variables)
    gradients3 = tape2.gradient(loss_value3, model.trainable_variables)
    # 二次梯度最后一层的b的梯度为None，需要剔除
    gradients2[-1] = 0
    gradients3[-1] = 0
    # 扰动计算梯度
    # temp = g.normal(fs[0].shape, stddev=1.0) + fs[0]
    # gradients4, loss_value4 = tune_per_step(temp)
    gradients = [gradients1[x] + gradients2[x] +gradients3[x] + si_coff * gradients5[x] for x in range(len(gradients2))]
    # 把梯度和变量进行绑定
    grads_and_vars = zip(gradients, model.trainable_variables)
    # 进行梯度更新
    my_opt.apply_gradients(grads_and_vars)
    loss_value = loss_value1 + loss_value2 + loss_value3 + loss_value5

    return loss_value, loss_value1, loss_value3,  loss_value2, loss_value5


def tune_per_step(temp):
    temp = tf.Variable(temp, name='x', dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            predicts = model(temp)
        # 计算神经网络的雅可比矩阵
        J_model = tape1.gradient(predicts, temp)
        loss_value2 = loss(J_model, 1.0)

    # 根据损失求梯度
    gradients2 = tape2.gradient(loss_value2, model.trainable_variables)
    vars_ = model.trainable_variables
    # 二次梯度最后一层的b的梯度为None，需要剔除
    gradients2[-1] = tf.constant(np.zeros(vars_[-1].shape), dtype=tf.float32)

    return gradients2, loss_value2


if __name__ == '__main__':
    import tensorflow as tf
    from pypower.case57 import case57
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    batch_size = 50
    si_coff = 0.1
    dir_case = './训练模型与损失/M4/'
    traning_data_dir = '../../DCOPF算例数据/57节点系统/负荷波动[20%]-新能源渗透率[34.28%]/case57_data_extension_13313.json'
    testing_data_dir = '../../DCOPF算例数据/57节点系统/负荷波动[20%]-新能源渗透率[34.28%]/case57_data_2000.json'
    testing_data_dir2 = '../../DCOPF算例数据/57节点系统/负荷波动[20%]-新能源渗透率[39.51%]/case57_data_2000.json'

    # 计算物理模型引导需要的参数
    mpc = case57()
    Aaj = np.zeros((mpc['bus'].shape[0], mpc['gen'].shape[0]))
    for i in range(mpc['gen'].shape[0]):
        Aaj[int(mpc['gen'][i, 0]-1), i] = 1

    temp_mpc = copy.deepcopy(mpc)
    temp_mpc = ext2int(temp_mpc)
    H = makePTDF(temp_mpc['baseMVA'], temp_mpc['bus'], temp_mpc['branch'])
    s_ = tf.constant(H[7, :][np.newaxis, :], dtype=tf.float32)
    s = np.dot(H, Aaj)
    s = tf.constant(s[7, :][np.newaxis, :], dtype=tf.float32)
    # 导入数据
    with open(traning_data_dir) as json_file:
        trainData = json.load(json_file)
        trainData = list(trainData.items())
    with open(testing_data_dir) as json_file:
        testData = json.load(json_file)
        testData = list(testData.items())

    with open(testing_data_dir2) as json_file:
        testData2 = json.load(json_file)
        testData2 = list(testData2.items())

    print("训练集：{}，测试集1：{}，测试集2：{}".format(len(trainData), len(testData), len(testData2)))

    # 搭建神经网络
    model = my_model()

    # 配置训练参数有优化器
    loss = tf.keras.losses.MeanAbsoluteError()
    my_opt = keras.optimizers.Adam(learning_rate=0.001, decay=2e-6)

    saver = tf.train.Checkpoint(optimizer=my_opt, model=model)
    model.compile(optimizer=my_opt, loss=loss)
    model.load_weights('./训练模型与损失/M1/Model.h5', by_name=False)

    # 设置训练损失值的保存
    if not os.path.exists(dir_case):
        os.makedirs(dir_case)
    f = open(dir_case + 'losses.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["epoch", "total_loss", "mse_loss", "gradient_loss", "dis_loss", "j_loss", "test1_loss", "test2_loss"])
    losses = []

    # 进行训练
    g = tf.random.get_global_generator()  # 随机扰动生成器
    for epoch in range(200):
        epoch_loss_avg_total = []
        epoch_loss_avg_mse = []
        epoch_loss_avg_g = []
        epoch_loss_avg_j = []
        epoch_loss_avg_br = []
        for fs in data_generator(trainData, batch_size=batch_size):
            loss_value, loss_value1, loss_value3, loss_value2, loss_value5= train_per_step(fs, g)
            epoch_loss_avg_total.append(loss_value.numpy())
            epoch_loss_avg_mse.append(loss_value1.numpy())
            epoch_loss_avg_g.append(loss_value3.numpy())
            epoch_loss_avg_j.append(loss_value2.numpy())
            epoch_loss_avg_br.append(loss_value5.numpy())

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

        csv_writer.writerow([epoch, np.mean(epoch_loss_avg_total), np.mean(epoch_loss_avg_mse), np.mean(epoch_loss_avg_g),
                             np.mean(epoch_loss_avg_j), np.mean(epoch_loss_avg_br),
                             np.mean(test_loss), np.mean(test2_loss)])
        print("Epoch {:03d}: Loss_total: {:.6f}, Loss_mse: {:.6f},Loss_g: "
              "{:.6f}, Loss_j: {:.6f}, Loss_br: {:.6f},test1_loss:{:.6f}, test2_loss:{:.6f} ".format(epoch,
                                                                    np.mean(epoch_loss_avg_total),
                                                                    np.mean(epoch_loss_avg_mse),
                                                                    np.mean(epoch_loss_avg_g),
                                                                    np.mean(epoch_loss_avg_j),
                                                                    np.mean(epoch_loss_avg_br),
                                                                    np.mean(test_loss),
                                                                    np.mean(test2_loss)))

    f.close()

    # 保存模型
    model.save('./{}/Model.h5'.format(dir_case))








