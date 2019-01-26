# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:42:55
# train.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''
# here put the import lib
#开始训练
import os
import numpy as np
import tensorflow as tf
from model import rnn_model
import sys
sys.path.append("program/tool")
from deal_data import process_poems
from until import generate_batch
 
 
def run_training(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    #产生诗向量，字映射表，字集合
    poems_vector, word_to_int, words = process_poems(FLAGS.file_path)
    
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
 
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    # 构建模型
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        words), rnn_size=FLAGS.rnn_size, num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)
    #tf里面提供模型保存的是tf.train.Saver()模块。检查checkpoint 内容最好的方法是使用Saver 加载它。
    saver = tf.train.Saver(tf.global_variables())
    #创建一个包含几个操作的op结点,当这个op结点运行完成，所有作为input的ops都被运行完成，这个操作没有返回值
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        #运行以上的节点
        sess.run(init_op)
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        # 从上次中断的checkpoint开始训练
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        file = open(FLAGS.log_file,'a')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                #输入的数据集切成多少块
                n_chunk = len(poems_vector) // FLAGS.batch_size
                print("n_chunk："+n_chunk)
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']#batches_inputs[n]代表取出第几块数据集
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                    file.write('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            # 如果Ctrl+c中断，保存checkpoint，
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))
            file.close()
        file.close()