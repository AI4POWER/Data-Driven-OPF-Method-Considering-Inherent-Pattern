"""
对traning_case30.py训练的模型进行测试
"""
# coding=utf-8
from unittest import case
import h5py, os
import numpy
import json
from pypower.rundcopf import rundcopf
from pypower.rundcpf import rundcpf
import numpy as np
from tensorflow import keras
import tensorflow as tf
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
from pypower.idx_brch import PF, QF, BR_STATUS, RATE_A, RATE_B, RATE_C, BR_R, BR_B, BR_X, T_BUS, F_BUS


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    batch_size = 1
    si_coff = 0.001
    thre = 1
    dir_case = './训练模型与损失/M5_old/'
    traning_data_dir = '../../DCOPF算例数据/负荷波动为[20%]-新能源渗透率[25.56%]/case30_data_10000.json'
    testing_data_dir = '../../DCOPF算例数据/负荷波动为[20%]-新能源渗透率[25.56%]/case30_data_2000.json'
    testing_data_dir2 = '../../DCOPF算例数据/负荷波动为[20%]-新能源渗透率[35.58%]/case30_data_2000.json'
    testing_data_dir3 = '../../DCOPF算例数据/负荷波动为[20%]-新能源渗透率[40.71%]/case30_data_2000.json'
    testing_data_dir4 = '../../DCOPF算例数据/负荷波动为[20%]-新能源渗透率[47.24%]/case30_data_2000.json'
    testing_data_dir5 = '../../DCOPF算例数据/负荷波动为[20%]-新能源渗透率[50.7%]/case30_data_2000.json'

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
    with open(testing_data_dir3) as json_file:
        testData3 = json.load(json_file)
        testData3 = list(testData3.items())
    with open(testing_data_dir4) as json_file:
        testData4 = json.load(json_file)
        testData4 = list(testData4.items())
    with open(testing_data_dir5) as json_file:
        testData5 = json.load(json_file)
        testData5 = list(testData5.items())

    print("训练集：{}，测试集1：{}，测试集2：{}，测试集3：{}，测试集4：{}，测试集5：{}".format(len(trainData),
                                                                                    len(testData),
                                                                                    len(testData2),
                                                                                    len(testData3),
                                                                                    len(testData4),
                                                                                    len(testData5)))

    # 搭建神经网络
    model = my_model()

    # 配置训练参数有优化器
    loss = tf.keras.losses.MeanAbsoluteError()
    my_opt = keras.optimizers.Adam(lr=0.001, decay=2e-6)

    saver = tf.train.Checkpoint(optimizer=my_opt, model=model)
    model.compile(optimizer=my_opt, loss=loss)
    model.load_weights('./{}/Model.h5'.format(dir_case), by_name=True)


    from pypower.case30 import case30
    mpc = case30()  # 导入case
    from pypower.ppoption import ppoption
    ppopt = ppoption(None, PF_DC=False, OPF_ALG_DC=200)
    ppopt['VERBOSE'] = 0
    ppopt['OUT_ALL'] = 0

    test_loss = []
    pf_error = []
    violate_rate = []
    testData = testData4
    pf1s = []
    pf2s = []
    pf_over = []
    for fs in data_generator(testData, batch_size=batch_size):
        predicts = model(fs[0])
        loss_value1 = np.abs(fs[1].numpy() - predicts.numpy())

        # DCPF计算线路潮流
        mpc['bus'][:, 2] = fs[0].numpy()[0, :] * 100
        mpc['gen'][:, 1] = fs[1].numpy()[0, :] * 100
        res = rundcpf(mpc, ppopt=ppopt)
        pf1 = res[0]['branch'][:, PF]

        mpc['gen'][:, 1] = predicts.numpy()[0, :] * 100
        res2 = rundcpf(mpc, ppopt=ppopt)
        pf2 = res2[0]['branch'][:, PF]

        pf_error.append(np.abs(pf1 - pf2))

        pf_limit = res2[0]['branch'][:, RATE_A] + 1

        violate_upper_limit = pf2 - pf_limit
        violate_upper_limit[violate_upper_limit > 0] = 1
        violate_upper_limit[violate_upper_limit <= 0] = 0

        violate_button_limit = pf2 + pf_limit
        violate_button_limit[violate_button_limit >= 0] = 0
        violate_button_limit[violate_button_limit < 0] = 1

        violate_rate.append(violate_upper_limit + violate_button_limit)
        pf1s.append(pf1)
        pf2s.append(pf2)

        temp = np.abs(pf2) - pf_limit
        temp[temp<0] = 0
        pf_over.append(temp)

        test_loss.append(loss_value1)

    print('线路越线条数：', np.sum(violate_rate))
    print('线路误差：', np.mean(pf_error))

    print('越线的平均量', np.sum(pf_over)/np.sum(violate_rate))

    violate_rate = np.array(violate_rate)
    pf1s = np.array(pf1s)
    pf2s = np.array(pf2s)
    pf_error = np.array(pf_error)
    pf_error[pf_error >= 1] = 1
    pf_error[pf_error < 1] = 0
    print('线路概率精度', np.mean(1 - pf_error))


#       越线条数    线路误差平均值   线路潮流概率精度    越线平均值
# M1:   14         0.64396MW     83.32%           3.40MW
# M2:   7          0.2226MW      95.40%           0.847MW
# M3:   4          0.5662MW      84.75%           1.41MW
# M4:   6          0.1574MW      96.68%           1.00MW
# M5:   3          0.0579MW      99.51%           0.756MW
