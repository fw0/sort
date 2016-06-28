import numpy as np
import tensorflow as tf
from collections import namedtuple
import pdb

class model(object):
    """
    variables in model:
    u_ts: embedding of hotel at position t
    s_ts: hidden state representing user's thoughts as they scroll down the hotel list
    a_ts: weight of s_ts when combining them to form c
    c: context used to make decision y based on p
    p: the probability of clicking (y=1)
    y: the binary decision
    """
    def add_placeholders(self):
        raise NotImplementedError

    def add_u_ts(self, *placeholders):
        raise NotImplementedError

    def add_s_ts(self, u_ts):
        raise NotImplementedError

    def add_a_ts(self, s_ts, u_ts):
        raise NotImplementedError

    def add_c(self, s_ts, a_ts):
        raise NotImplementedError

    def add_score_and_p(self, c):
        raise NotImplementedError

    def add_loss(self, y, p):
        raise NotImplementedError


    
class config(object):

    initializer = tf.contrib.layers.xavier_initializer

    batch_size = 8
    
    x_dim = 5
    num_steps = 4
    u_dim = 3
    s_dim = 2
    num_classes = 2

    lr = 0.001

    reg = 1.0

class basic_model(model):

    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.u_ts = self.add_u_ts()
        self.s_ts = self.add_s_ts(self.u_ts)
        self.a_ts = self.add_a_ts(self.s_ts, self.u_ts)
        self.c = self.add_c(self.a_ts, self.s_ts)
        self.score, self.p = self.add_score_and_p(self.c)
        self.loss = self.add_loss(self.label_ph, self.score)
        self.regularization = self.add_regularization()
        self.training_op = self.add_training_op(self.loss, self.regularization)

    def set_data(self, data_iterator):
        self.data_iterator = data_iterator
        
    def add_placeholders(self):
        self.input_ph = [tf.placeholder(tf.float32, shape=[None, self.config.x_dim]) for i in xrange(self.config.num_steps)]
        self.label_ph = tf.placeholder(tf.float32, shape=[None, 2])

    def add_u_ts(self):
        u_ts = [None for i in xrange(self.config.num_steps)]
        with tf.variable_scope('u', initializer = self.config.initializer()):
            self.W_xu = tf.get_variable('W_xu', shape=[self.config.x_dim, self.config.u_dim])
            for i in xrange(self.config.num_steps):
                u_ts[i] = tf.matmul(self.input_ph[i], self.W_xu)
        return u_ts

    def add_s_ts(self, u_ts):
        s_ts = [None for i in xrange(self.config.num_steps)]
        with tf.variable_scope('s', initializer = self.config.initializer()):
            self.W_ss = tf.get_variable('W_ss', [self.config.s_dim, self.config.s_dim])
            self.W_us = tf.get_variable('W_us', [self.config.u_dim, self.config.s_dim])
            self.b_s = tf.get_variable('b_s', [1, self.config.s_dim])
            for i in xrange(self.config.num_steps):
                if i != 0:
                    s_ts[i] = tf.sigmoid(tf.matmul(s_ts[i-1], self.W_ss) + tf.matmul(u_ts[i], self.W_us) + self.b_s)
                elif i == 0:
                    s_ts[i] = tf.sigmoid(tf.matmul(tf.ones((self.config.batch_size, self.config.s_dim)), self.W_ss) + tf.matmul(u_ts[i], self.W_us) + self.b_s)
                else:
                    assert False
        return s_ts

    def add_a_ts(self, s_ts, u_ts):
        with tf.variable_scope('a', initializer = self.config.initializer()):
            self.w_sa = tf.get_variable('w_sa', [self.config.s_dim, 1])
            self.w_ua = tf.get_variable('w_ua', [self.config.u_dim, 1])
            eps = [None for i in xrange(self.config.num_steps)]
            for i in xrange(self.config.num_steps):
                eps[i] = tf.matmul(s_ts[i], self.w_sa) + tf.matmul(u_ts[i], self.w_ua)
            return tf.nn.softmax(tf.concat(1, eps))

    def add_c(self, a_ts, s_ts):
        with tf.variable_scope('c', initializer = self.config.initializer()):
            self.c = tf.reduce_sum(tf.mul(tf.concat(2, [tf.expand_dims(s_t,-1) for s_t in self.s_ts]), tf.expand_dims(self.a_ts, 1)), reduction_indices=[2])
        return self.c

    def add_score_and_p(self, c):
        with tf.variable_scope('p', initializer = self.config.initializer()):
            self.w_cp = tf.get_variable('w_cp', [self.config.s_dim, self.config.num_classes])
            score = tf.matmul(c, self.w_cp)
        return score, tf.nn.softmax(score)

    def add_loss(self, y, p):
        loss = tf.nn.softmax_cross_entropy_with_logits(p, self.label_ph)
        return loss

    def add_regularization(self):
        return self.config.reg * sum(map(tf.nn.l2_loss, [self.W_xu, self.W_ss, self.W_us, self.b_s, self.w_sa, self.w_ua, self.w_cp]))
    
    def add_training_op(self, loss, regularization):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        training_op = optimizer.minimize(loss + regularization)
        return training_op

    def run_epoch(self, session, data_iterator):
        losses = []
        for (i,(x,y)) in enumerate(data_iterator):
            feed = {self.input_ph[i]: x[i] for i in xrange(self.config.num_steps)}
            feed[self.label_ph] = y
            loss, _ = session.run([self.loss, self.training_op], feed_dict = feed)
            print 'step %d loss: %.2f' % (i, np.mean(loss))
        return np.mean(loss)
