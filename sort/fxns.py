import numpy as np
import tensorflow as tf

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
    def add_placeholders(self)
        raise NotImplementedError

    def add_u_ts(self, *placeholders):
        raise NotImplementedError

    def add_s_ts(self, u_ts):
        raise NotImplementedError

    def add_a_ts(self, s_ts, u_ts):
        raise NotImplementedError

    def add_c(self, s_ts, a_ts):
        raise NotImplementedError

    def add_p(self, c):
        raise NotImplementedError

    def add_loss(self, y, p):
        raise NotImplementedError

class config(object):

    initializer = tf.contrib.layers.xavier_initializer()

    x_dim = 50
    num_steps = 25
    u_dim = 100
    s_dim = 25

    lr = 0.001

class basic_model(model):

    def __init__(self, config, data_iterator):
        self.config, self.data_iterator = config, data_iterator

    def add_placeholders(self, x_ts, y_ts):
        self.input_ph = [tf.placeholder(tf.float32, shape=[None, self.config.x_dim]) for i in xrange(self.config.num_steps)]
        self.label_ph = tf.placeholder(tf.float32, shape=[None])

    def add_u_ts(self, input_ph):
        self.u_ts = [None for i in xrange(self.config.num_steps)]
        with tf.variable_scope('u', initializer = self.config.initializer):
            self.W_xu = tf.get_variable('W_xu', [self.config.x_dim, self.config.u_dim])
            for i in xrange(xrange(self.config.num_steps)):
                self.u_ts[i] = tf.matmul(self.input_ph[i], self.W_xu)
        return self.u_ts

    def add_s_ts(self, u_ts):
        self.s_ts = [None for i in xrange(self.config.num_steps)]
        with tf.variable_scope('s', initializer = self.config.initializer):
            self.W_ss = tf.get_variable('W_ss', [self.config.s_dim, self.config.s_dim])
            self.W_us = tf.get_variable('W_us', [self.config.u_dim, self.config.s_dim])
            self.b_s = tf.get_variable('b_s', [self.config.s_dim])
            for i in xrange(self.config.num_steps):
                if i != 0:
                    self.s_ts[i] = tf.sigmoid(tf.matmul(self.s_ts[i-1], self.W_ss) + tf.matmul(self.u_ts[i], self.W_us) + self.b_s)
                elif i == 0:
                    self.s_ts[i] = tf.sigmoid(tf.matmul(tf.ones(self.config.batch_size, self.config.s_dim), self.W_ss) + tf.matmul(self.u_ts[i], self.W_us) + self.b_s)
                else:
                    assert False
        return self.s_ts

    def add_a_ts(self, s_ts):
        with tf.variable_scope('a', initializer = self.config.initializer):
            self.w_sa = tf.get_variable('w_sa', [self.config.s_dim])
            self.w_ua = tf.get_variable('w_ua', [self.config.s_dim])
            eps = [None for i in self.config.num_steps]
            for i in self.config.num_steps:
                eps[i] = tf.matmul(self.s_ts[i], self.w_sa) + tf.matmul(self.w_ua, self.w_ua)
            self.a_ts = tf.nn.softmax(tf.concat(1, eps))
        return self.a_ts

    def add_c(self, a_ts, s_ts):
        with tf.variable_scope('c', initializer = self.config.initializer):
            self.c = tf.reduce_sum(tf.mul(tf.concat(2, [tf.expand_dims(s_t,-1) for s_t in self.s_ts]), tf.expand_dims(self.a_ts,-1)), reduction_indices=[2])
        return self.c

    def add_p(self, c):
        with tf.variable_scope('p', initializer = self.config.initializer):
            self.w_cp = tf.get_variable('w_cp', [self.config.s_dim])
            self.p = tf.matmul(c, self.w_cp)
        return self.p

    def add_loss(self, y, p):
        zero_p = tf.ones(self.batch_size) - self.p
        mat_p = tf.transpose(tf.pack([zero_p,p]))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(mat_p, self.label_ph)
        return self.loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.calculate_loss)
        return self.train_op

    def run_epoch(self, session, data_iterator):
        losses = []
        for (i,(x,y)) in enumerate(data_iterator(self.config.batch_size)):
            feed = {self.input_ph: x, self.label:ph: y}
            loss, _ = session.run[self.loss, self.train_op]
            print 'step %d loss: %.2f' % (i, loss)
        return np.mean(losses)
