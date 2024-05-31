
# coding=utf-8
from unittest import case
import h5py, os
import numpy
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt


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


def my_model():
    node_num = 57
    input_PD = keras.Input(shape=(node_num,), name='input_PD')
    characteristic = keras.layers.Dense(units=200, activation='relu')(input_PD)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    PG_out = keras.layers.Dense(units=7, name='output_PG')(characteristic)
    # outs = Last_Layer(W, B, name='outputs')(PG_out)

    model = keras.Model(inputs=input_PD, outputs=[PG_out, characteristic])
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


def AutoEC():
    node_num = 57
    input_PD = keras.Input(shape=(node_num,), name='input_PD0')
    characteristic = keras.layers.Dense(units=200, activation='relu')(input_PD)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    characteristic = keras.layers.Dense(units=200, activation='relu')(characteristic)
    PG_out = keras.layers.Dense(units=57, name='output_PG')(characteristic)
    # outs = Last_Layer(W, B, name='outputs')(PG_out)

    model = keras.Model(inputs=input_PD, outputs=PG_out)
    model.summary()
    # model.compile()

    return model


if __name__ == '__main__':
    import tensorflow as tf
    from pypower.case30 import case30
    import os

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    batch_size = 50
    si_coff = 0.001
    thre = 5
    thre_encoder = 5
    testing_data_dir1 = '../../../DCOPF算例数据/57节点系统/负荷波动[20%]-新能源渗透率[34.28%]/case57_data_extension_13313.json'
    testing_data_dir2 = '../../../DCOPF算例数据/57节点系统/负荷波动[20%]-新能源渗透率[45.96%]/case57_data_2000.json'

    # 导入数据
    with open(testing_data_dir1) as json_file:
        testData1 = json.load(json_file)
        testData1 = list(testData1.items())
    with open(testing_data_dir2) as json_file:
        testData2 = json.load(json_file)
        testData2 = list(testData2.items())

    # # 检测测试集是否符合要求
    # encoder = AutoEC()
    # encoder.load_weights('./训练模型与损失/Reconstruct_grad/Model.h5', by_name=True)

    # 搭建神经网络
    model = my_model()

    # 配置训练参数有优化器
    loss = tf.keras.losses.MeanAbsoluteError()
    my_opt = keras.optimizers.Adam(learning_rate=0.001, decay=2e-6)

    saver = tf.train.Checkpoint(optimizer=my_opt, model=model)
    model.compile(optimizer=my_opt, loss=loss)
    solving_model ="../训练模型与损失/M5"
    model.load_weights('{}/Model.h5'.format(solving_model), by_name=False)

    testData1_ = testData2
    traindata = []
    for i in range(len(testData1_)):
        # 误差计算
        fs = [np.array(testData1_[i][1]['pd']).T, np.array(testData1_[i][1]['pg']).T]
        predicts = model(fs[0])
        features = predicts[1].numpy()
        features = np.concatenate([features, fs[0]], axis=1)[0]
        traindata.append(features)

    traindata = np.array(traindata)
    np.savez('Testdata2_M5.npz', traindata=traindata)













