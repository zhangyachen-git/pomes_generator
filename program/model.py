# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:41:18
# model.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''
# here put the import lib
# 设置模型
import tensorflow as tf
import numpy as np

tf.reset_default_graph()


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:len(words)=6110
    :param rnn_size:128
    :param num_layers:2
    :param batch_size:64
    :param learning_rate:0.01
    :return:
    """
    end_points = {}  # 终点
    # 构建RNN基本单元RNNcell
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.LSTMCell
    # cell:<class 'tensorflow.python.ops.rnn_cell_impl.LSTMCell'>
    cell = cell_fun(rnn_size, state_is_tuple=True)
    # 构建堆叠rnn，这里选用两层的rnn
    # cell:<tensorflow.python.ops.rnn_cell_impl.LSTMCell object at 0x7fc7759a23c8>
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # 如果是训练模式，output_data不为None，则初始状态shape为[batch_size * rnn_size]
    # 如果是生成模式，output_data为None，则初始状态shape为[1 * rnn_size]
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)
    # 构建隐层
    with tf.device("/cpu:0"):
        # <tf.Variable 'embedding:0' shape=(6111, 128) dtype=float32_ref>
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            # 返回-1和1之间均匀分布的（vocab_size + 1）*rnn_size矩阵，列由128维表示，每一字构成一行，行数就是字对应的word_int映射
            [vocab_size + 1, rnn_size], -1.0, 1.0))  # 返回-1和1之间均匀分布的（vocab_size + 1）*rnn_size矩阵
        # input_data是二维矩阵代表某一块batch数据集，每一行代表一首诗，由poem_vec表示，
        # 通过遍历每首诗的每个字，字的表示是word_int映射，找到了该映射数字也就找到了在embedding对应的行数
        # <tf.Tensor 'embedding_lookup/Identity:0' shape=(64, ?, 128) dtype=float32>
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    # [batch_size, ?, rnn_size] = [64, ?, 128]
    # outputs:<tf.Tensor 'rnn/transpose_1:0' shape=(64, ?, 128) dtype=float32>
    # last_state:(LSTMStateTuple(c=<tf...=float32>), LSTMStateTuple(c=<tf...=float32>))
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    # 只有output的列数和weights的行数相等才能做相乘
    # output:<tf.Tensor 'Reshape:0' shape=(?, 128) dtype=float32>
    output = tf.reshape(outputs, [-1, rnn_size])
    # weights:<tf.Variable 'Variable:0' shape=(128, 6111) dtype=float32_ref>
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))  # 产生正态分布做权重
    # blas:<tensorflow.python.ops.rnn_cell_impl.MultiRNNCell object at 0x7fc7757ef358>
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))  # 产生偏置的个数与weights的列数相等
    # logits:<tf.Tensor 'BiasAdd:0' shape=(?, 6111) dtype=float32>
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)  # 矩阵相乘
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        # labels:<tf.Tensor 'one_hot:0' shape=(?, 6111) dtype=float32>
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size+1]
        # loss:<tf.Tensor 'softmax_cross_entropy_with_logits_sg/Reshape_2:0' shape=(?,) dtype=float32>
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        # total_loss:<tf.Tensor 'Mean:0' shape=() dtype=float32>
        total_loss = tf.reduce_mean(loss)
        # train_op:<tf.Operation 'Adam' type=NoOp>
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points
