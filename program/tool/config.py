# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:18:24
# config.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''
# here put the import lib
import os
import tensorflow as tf

# tf.app.flags.DEFINE_xxx()就是添加命令行的optional argument（可选参数）
# os.path.abspath(path)：返回path规范化的绝对路径。
# epoch：1个epoch等于使用训练集中的全部样本训练一次
# 运行程序设置
tf.app.flags.DEFINE_string('run_flag', '1', 'run train flag: 0;run test flag:1')
# 文件信息
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('../data/output/model'), 'model save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('../data/original_data/poetry.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('log_path', os.path.abspath('../project_doumentation/train_log/train_log.txt'),
                           'file name of log.')
tf.app.flags.DEFINE_string('tensorflow_logs', os.path.abspath('../project_doumentation/train_log/tensorflow_logs'),
                           'file name of log.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')

# 参数信息
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')
tf.app.flags.DEFINE_integer('rnn_size', 128, 'rnn size.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'num layers.')
# f.app.flags.FLAGS可以从对应的命令行参数取出参数。
FLAGS = tf.app.flags.FLAGS
