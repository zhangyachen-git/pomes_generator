# -*- coding: utf-8 -*-
'''
# Created on 2019/01/26 10:45:55
# main.py
# @author: ZhangYachen
#@Version :   1.0
#@Contact :   aachen_z@163.com
'''
# here put the import lib
import train
import test
from tool.config import FLAGS

if __name__ == '__main__':

    if FLAGS.run_flag == "0":
        train.run_training(FLAGS)
    else:
        begin_char = input('## 请输入第一个字:')
        poem = test.gen_poem(FLAGS, begin_char)
        test.pretty_print_poem(poem_=poem)
