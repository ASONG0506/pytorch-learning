# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 2019-11-12
# usage: 借助于tensorflow的logger文件机制，写log，从而记录训练的过程
# --------------------

import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        """
        指定log的路径
        :param log_dir:
        """
        self.writer = tf.Summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        记录scalar
        :param tag:
        :param value:
        :param step:
        :return:
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag = tag, simple_value = value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """
        记录scalar
        :param tag_value_pairs:
        :param step:
        :return:
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value = value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)