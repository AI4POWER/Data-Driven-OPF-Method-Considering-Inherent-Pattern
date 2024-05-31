
# coding=utf-8
from unittest import case
import h5py, os
import numpy
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf
import random
import csv


def data_generator(data, batch_size=15):
    idx = np.random.permutation(len(data))

    for k in range(int(np.ceil(len(data) / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        idx_ = idx[from_idx:to_idx]
        inputs_ = data[idx_]

        yield tf.Variable(inputs_[:, :200], name='x', dtype=tf.float32), \
              tf.constant(inputs_[:, 200:], dtype=tf.float32)


def my_model():
    node_num = 200
    input_PD = keras.Input(shape=(node_num,), name='input_PD')
    characteristic = keras.layers.Dense(units=200, activation='relu')(input_PD)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    PG_out = keras.layers.Dense(units=57, name='output_PG')(characteristic)
    # outs = Last_Layer(W, B, name='outputs')(PG_out)

    model = keras.Model(inputs=input_PD, outputs=PG_out)
    model.summary()
    # model.compile()

    return model


def train_per_step(fs, g):
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            predicts = model(fs[0])
            # pg的有监督loss
            loss_value1 = loss(fs[1], predicts)
        # # 计算神经网络的雅可比矩阵
        # J_model3 = tape1.gradient(predicts, fs[0])
        # loss_value3 = loss(J_model3, 1.0)

    # 根据损失求梯度
    gradients1 = tape2.gradient(loss_value1, model.trainable_variables)
    # gradients3 = tape2.gradient(loss_value3, model.trainable_variables)
    # 二次梯度最后一层的b的梯度为None，需要剔除
    # gradients3[-1] = 0
    # 扰动计算梯度
    # temp = g.normal(fs[0].shape, stddev=1.0) + fs[0]
    # gradients4, loss_value4 = tune_per_step(temp)
    # 把梯度和变量进行绑定
    grads_and_vars = zip(gradients1, model.trainable_variables)
    # 进行梯度更新
    my_opt.apply_gradients(grads_and_vars)
    loss_value = loss_value1

    return loss_value


if __name__ == '__main__':
    import tensorflow as tf
    from pypower.case30 import case30
    import os

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    batch_size = 500
    si_coff = 0.001
    dir_case = './训练模型与损失/Reconstruct_hidden/'
    training_data_dir = './Traindata_M5.npz'
    testing_data_dir1 = './Testdata1_M5.npz'
    testing_data_dir2 = './Testdata2_M5.npz'

    # 导入数据
    data = np.load(training_data_dir, allow_pickle=True)
    traindata = data['traindata']

    data = np.load(testing_data_dir1, allow_pickle=True)
    testdata1 = data['traindata']

    data = np.load(testing_data_dir2, allow_pickle=True)
    testdata2 = data['traindata']

    # 搭建神经网络
    model = my_model()

    # 配置训练参数有优化器
    loss = tf.keras.losses.MeanSquaredError()  # 'categorical_crossentropy' #MeanAbsoluteError()
    my_opt = keras.optimizers.Adam(learning_rate=0.001, decay=2e-6)

    saver = tf.train.Checkpoint(optimizer=my_opt, model=model)
    model.compile(optimizer=my_opt, loss=loss)

    # 设置训练损失值的保存
    if not os.path.exists(dir_case):
        os.makedirs(dir_case)
    f = open(dir_case + 'losses.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["epoch", "total_loss", "test_loss"])   # 构建列表头
    losses = []

    # 进行训练
    g = tf.random.get_global_generator()  # 随机扰动生成器
    for epoch in range(1000):
        epoch_loss_avg = []
        test_loss1 = []
        test_loss2 = []
        for fs in data_generator(traindata, batch_size=batch_size):
            loss_value = train_per_step(fs, g)
            epoch_loss_avg.append(loss_value.numpy())

        for fs in data_generator(testdata1, batch_size=batch_size):
            predicts = model(fs[0])
            loss_value1 = loss(fs[1], predicts)
            test_loss1.append(loss_value1)
        for fs in data_generator(testdata2, batch_size=batch_size):
            predicts = model(fs[0])
            loss_value1 = loss(fs[1], predicts)
            test_loss2.append(loss_value1)

        csv_writer.writerow([epoch, np.mean(epoch_loss_avg), np.mean(test_loss1), np.mean(test_loss2)])
        print("Epoch {:03d}: Loss: {:.6f}, TestError1:{:.6f},  TestError2:{:.6f}".format(epoch,
                                                                                         np.mean(epoch_loss_avg),
                                                                                         np.mean(test_loss1),
                                                                                         np.mean(test_loss2)
                                                                                         ))

    f.close()
    model.save('./{}/Model.h5'.format(dir_case))








