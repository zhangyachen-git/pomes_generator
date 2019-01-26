# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:22:54
# until.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''
# here put the import lib
import numpy as np


def generate_batch(batch_size, poems_vec, word_to_int):
    #总共切多少块,batch_size如果取64，取整数原则34646//64=541块
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        #batches为二维数组，每块固定有多少首诗，长度为batch_size
        batches = poems_vec[start_index:end_index]
        #确定该batches中字数最多的那首诗的字数
        length = max(map(len, batches))
        #每个x_data由空格组成一个batch_size*length的矩阵
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        #row代表第几首诗，每首诗都有length的长度，从第一个字的位置开始由batch来还原，填不满的就空着。
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch
        #复制过来后，x_data的变化不会引起y_data变化
        y_data = np.copy(x_data)
        #y_data代表目标输出，由x_data向左移动形成，最后一列没有意义，给出每首诗的上一个字预测一下个字，直到预测完整首诗
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data                y_data
        第一首：[6,2,4,6,9]      第一首：[2,4,6,9,9]
        第二首：[1,4,2,8,5]      第二首：[4,2,8,5,5]
        """
        #x_batches 和y_batches的长度均为541，即储存着541块数据子集
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches