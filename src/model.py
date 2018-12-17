import tensorflow as tf
import numpy as tf
from config import *


class Network:

    def __init__(self, scope):
        self.embedding_dim = Embedding_dim
        self.learning_rate = Learning_rate
        self.scope = scope
        self.filters = CNN_Filters
        self.kernel_size = CNN_Kernel_size
        self.strides = CNN_Strides
        self.every_steps_save = Every_steps_save
        self.model_load_path = Model_load_path

        self.training_step = 0

        # define session
        conf = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)

        self.initialize_ph()
        self.initialize_network()

        # define the saver
        self.saver = tf.train.Saver(max_to_keep=3)
        if Model_load:
            self.model_load()
        else:
            self.sess.run(tf.initialize_all_variables())

    def initialize_ph(self):
        self.input_ph = tf.placeholder(tf.float32, shape=[
                                       None, 1, Max_sentence_length, Embedding_dim], name="sentence_input")
        self.target = tf.placeholder(
            tf.float32, shape=[None, ], name="standard_out")

    def initialize_network(self):

        out = self.input_ph

        # cnn part
        with tf.variable_scope("cnn_part" + self.scope)
            self.conv1 = tf.layers.conv2d(
                inputs=out,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=None,
                padding="valid"
            )
            self.conv1 = tf.nn.sigmoid(tf.nn.pool(self.conv1, window_shape=, pooling_type="MAX", strides=, padding=))

            self.conv2 = tf.layers.conv1d(
                inputs=self.conv1,
                filters=,
                kernel_size=,
                strides=,
                activation=None,
                padding="valid"
            )
            self.conv2 = tf.nn.sigmoid(tf.nn.pool(self.conv2, window_shape=, pooling_type="MAX", strides=, padding=))

            self.conv3 = tf.layers.conv1d(
                inputs=self.conv2,
                filters=,
                kernel_size=,
                strides=,
                activation=None,
                padding="valid"
            )
            self.conv3 = tf.nn.sigmoid(tf.nn.pool(self.conv3, window_shape=, pooling_type="MAX", strides=, padding=))

            self.conv4 = tf.layers.conv1d(
                inputs=self.conv3,
                filters=,
                kernel_size=,
                strides=,
                activation=None,
                padding="valid"
            )
            self.conv4 = tf.nn.sigmoid(self.conv4)

        self.cnn_out = self.conv4

        # lstm
        with tf.variable_scope("lstm_part" + self.scope)
            self.lstm = tf.nn.rnn_cell.LSTMCell(
                num_units=,
                use_peepholes=True,
                initializer=tf.contrib.layers.initializers.xavier_initializer(),
                num_proj=,
                name="lstm_cell"
            )

            self.lstm_out, state = tf.nn.dynamic_rnn(
            )

        self.fc_input = tf.concat([self.cnn_out, self.lstm_out])

        # fc
        with tf.variable_scope("fc_part" + self.scope):
            fc_out = self.fc_input
            for output_num in DNN_Shape:
                layer = tf.layers.dense(
                    inputs=fc_out,
                    units=output_num,
                    activation=tf.nn.sigmoid
                )

        self.nn_output = fc_out

        self.loss = tf.reduce_mean(tf.square(self.nn_output - self.target))
        self.trainer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def train(self, data):

        self.training_step = self.training_step + 1

        _, loss = self.sess.run([self.trainer, self.loss], feed_dict={
            self.input_ph: data['words'],
            self.standard_out: data['tags']
        })

        if self.training_step % self.every_steps_save == 1:
            self.model_save()

    def get_result(self, data):

        output = self.sess.run([self.nn_output], feed_dict={
            self.input_ph: data['words'],
            self.standard_out: data['tags']
        })

    def model_save(self, name=None):

        print("now training step %d...model saving..." % (self.training_step))
        if name == None:
            self.saver.save(self.sess, "model/training_step" + self.scope,
                            global_step=self.training_step)
        else:
            self.saver.save(self.sess, name)

    def model_load(self):

        self.saver.restore(self.sess, self.model_load_path)