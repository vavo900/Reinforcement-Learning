import numpy as np
from algo import ValueFunctionWithApproximation
import tensorflow as tf
from tensorflow import keras


class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method

        self.state_dims = state_dims
        self.X = tf.placeholder(tf.float32, shape=[1,2], name="X")
        # self.X = tf.placeholder("float", [None, state_dims])
        self.Y = tf.placeholder(tf.float32, name="Y")
        # self.Y = tf.placeholder("float")

        self.weights = {
            'h1': tf.Variable(tf.random_normal([state_dims, 32])),
            'h2': tf.Variable(tf.random_normal([32, 32])),
            'out': tf.Variable(tf.random_normal([32, 1]))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([32])),
            'b2': tf.Variable(tf.random_normal([32])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        self.Y_hat = self.neural_net(self.X)
        self.loss_op = 0.5 * tf.losses.mean_squared_error(self.Y, self.Y_hat)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,
                                           beta2=0.999)
        self.train_op = optimizer.minimize(self.loss_op)
        self.init = tf.global_variables_initializer()
        # self.epoch = 2000

        self.sess = tf.Session()
        self.sess.run(self.init)

    def __call__(self, s):
        # TODO: implement this metho
        s = s.reshape(1,2)
        pred = self.sess.run(self.Y_hat, feed_dict={self.X: s})
        pred = pred.item()
        return pred

    def update(self, alpha, G, s_tau):
        # TODO: implement this method

        # for i in range(0, epoch):
        s_tau = s_tau.reshape(1,2)
        self.sess.run(self.train_op, feed_dict={self.X: s_tau, self.Y: G})
        # loss = self.sess.run(self.loss_op, feed_dict={self.X: s_tau, self.Y: G})
        # # print("s_tau, G, loss", s_tau, G, loss)
        # pred = self.sess.run(self.Y_hat, feed_dict={self.X: s_tau})
        # # print("s_tau, G, pred", s_tau, G, pred)

        return None

    def neural_net(self, X):
        # hidden layer 1
        layer_1 = tf.add(tf.matmul(X, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)  # activation
        # hidden layer 2
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)  # activation
        # output layer
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

