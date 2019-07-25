# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:42:55
# train.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''
# here put the import lib
# 开始训练
import os
import tensorflow as tf
from model import rnn_model
from tool.deal_data import process_poems
from tool.until import generate_batch


def run_training(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    # 产生诗向量，字映射表，字集合
    # pems_vector:[[2, 50, 179, 394, 1081, 597, 0, 13, 351, ...], [2, 161, 343, 973, 30, 2455, 0, 155, 297, ...], [2, 10, 89, 13, 644, 1085, 0, 340, 33, ...], [2, 644, 1081, 689, 204, 1116, 0, 25, 1242, ...], [2, 50, 259, 2401, 58, 137, 0, 25, 163, ...], [2, 6, 346, 25, 87, 106, 0, 439, 1926, ...], [2, 2951, 263, 1917, 28, 437, 0, 1155, 219, ...], [2, 238, 173, 873, 847, 552, 0, 1680, 175, ...], [2, 609, 1363, 17, 155, 56, 0, 915, 839, ...], [2, 202, 650, 79, 685, 3330, 0, 780, 539, ...], [2, 13, 4887, 1198, 2139, 699, 0, 1150, 2553, ...], [2, 107, 2126, 1745, 1103, 16, 0, 202, 1399, ...], [2, 89, 119, 296, 768, 173, 0, 200, 187, ...], [2, 539, 764, 630, 17, 386, 0, 435, 816, ...], ...]
    # word_to_int:{' ': 6109, '2': 5534, 'B': 2, 'E': 3, 'F': 5355, 'p': 5354, 'ē': 5531, 'ń': 5584, '□': 705, '、': 4110, '。': 1, '】': 5023, '一': 10, '丁': 1484, ...}
    # word:('，', '。', 'B', 'E', '不', '人', '山', '风', '日', '无', '一', '云', '花', '春', ...)
    poems_vector, word_to_int, words = process_poems(FLAGS.file_path)
    # baches_inputs:[array([[   2,   50, ...ype=int32), array([[   2,   75, ...ype=int32), array([[   2,   53, ...ype=int32), array([[   2,  121, ...ype=int32), array([[   2,  357, ...ype=int32), array([[   2,  168, ...ype=int32), array([[   2, 2483, ...ype=int32), array([[   2, 2501, ...ype=int32), array([[   2,  283, ...ype=int32), array([[   2,   80, ...ype=int32), array([[   2, 2204, ...ype=int32), array([[   2,  929, ...ype=int32), array([[   2,  671, ...ype=int32), array([[   2, 1440, ...ype=int32), ...]
    # batches_outputs:[array([[  50,  179, ...ype=int32), array([[  75,   35, ...ype=int32), array([[  53,  963, ...ype=int32), array([[ 121,  328, ...ype=int32), array([[ 357,  283, ...ype=int32), array([[ 168,   21, ...ype=int32), array([[2483,  641, ...ype=int32), array([[2501, 2176, ...ype=int32), array([[ 283,   11, ...ype=int32), array([[  80,  846, ...ype=int32), array([[2204, 2024, ...ype=int32), array([[ 929, 2500, ...ype=int32), array([[ 671, 1429, ...ype=int32), array([[1440,  211, ...ype=int32), ...]
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
    # inpuy_data:<tf.Tensor 'Placeholder:0' shape=(64, ?) dtype=int32>
    # output_targets:<tf.Tensor 'Placeholder_1:0' shape=(64, ?) dtype=int32>
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    # 构建模型
    # end_points:{'initial_state': (LSTMStateTuple(c=<tf...=float32>), LSTMStateTuple(c=<tf...=float32>)), 'last_state': (LSTMStateTuple(c=<tf...=float32>), LSTMStateTuple(c=<tf...=float32>)), 'loss': <tf.Tensor 'softmax_...e=float32>, 'output': <tf.Tensor 'Reshape:...e=float32>, 'total_loss': <tf.Tensor 'Mean:0' ...e=float32>, 'train_op': <tf.Operation 'Adam'...type=NoOp>}
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        words), rnn_size=FLAGS.rnn_size, num_layers=FLAGS.num_layers, batch_size=FLAGS.batch_size,
                           learning_rate=FLAGS.learning_rate)
    # tf里面提供模型保存的是tf.train.Saver()模块。检查checkpoint 内容最好的方法是使用Saver 加载它。
    # saver:<tensorflow.python.training.saver.Saver object at 0x7fc7756eaf28>
    saver = tf.train.Saver(tf.global_variables())
    # 创建一个包含几个操作的op结点,当这个op结点运行完成，所有作为input的ops都被运行完成，这个操作没有返回值
    # inti_op:<tf.Operation 'group_deps' type=NoOp>
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # 写入tensorflow日志
        writer = tf.summary.FileWriter(FLAGS.tensorflow_logs, sess.graph)
        # 运行以上的节点
        sess.run(init_op)
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        # 从上次中断的checkpoint开始训练
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('## start training...')
        file = open(FLAGS.log_path, 'a')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                # 输入的数据集切成多少块
                # n_chunk = len(poems_vector) // FLAGS.batch_size
                n_chunk = 50
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']  # batches_inputs[n]代表取出第几块数据集
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))
                    file.write('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss) + "\n")
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            # 如果Ctrl+c中断，保存checkpoint，
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))
            writer.close()
            file.close()
    file.close()
    writer.close()
