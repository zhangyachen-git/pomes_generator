# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:16:32
# deal_data.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''

'''
数据处理
'''
# here put the import lib

import collections
import numpy as np
 
start_token = 'B'#begin
end_token = 'E'#end
#数据集：总共有34646首诗,1721655个字(6110个去重后的字)
 
def process_poems(file_name):
    # poems -> list of numbers诗集
    poems = []#是二维数组，但不是矩阵，因为每首诗的长度不同
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')#以冒号来分开
                content = content.replace(' ', '')#去掉所有的空格
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:#去掉特殊符号
                    continue
                if len(content) < 5 or len(content) > 79:#内容少于5个字或大于79个字为异常诗需要剔除，跳出本次循环
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    #取出所有诗中所有的字构成一维数组,比如['低','头','思','故','乡','低',]
    all_words = [word for poem in poems for word in poem]
    #以字为key，该字出现的次数为value形成字典,按value从大到小排列{'不'：6000,'的'：5800,}
    counter = collections.Counter(all_words)
    #对计数结果进行由大到小排序，返回的结果是一个数组，数组元素为(某个字，该字的次数),[('不'：6000),('的'：5800),]
    count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    #按照出现次数由大到小的顺序取出所有的字放到一个小括号中，字与字之间用逗号隔开,('不','的',)
    words, _ = zip(*count_pairs)
    #末尾加一个空格,('不','的', , ,' ')
    words = words + (' ',)
    #为每个字打上位置标签，从0开始，形成字典,高频次的字在前面{'不'：0,'的':1, , ,' ':6110}
    word_int_map = dict(zip(words, range(len(words))))
    #每首诗中的字都能在word_int_map中找到位置标签，poems_vector是二维矩阵，行数为诗的个数，列代表每首诗的字的位置标签
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]
 
    return poems_vector, word_int_map, words


file_name = 'data/original_data/poetry.txt'
poems_vector, word_int_map, words = process_poems(file_name)
