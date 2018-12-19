import tensorflow as tf
import numpy as np
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
                                       None, Max_sentence_length, Embedding_dim, 1], name="sentence_input")
        self.target = tf.placeholder(
            tf.float32, shape=[None, 1], name="standard_out")

    def initialize_network(self):

        out = self.input_ph

        # cnn part
        with tf.variable_scope("cnn_part" + self.scope):
            self.conv1 = tf.layers.conv2d(
                inputs=out,
                filters=256,
                kernel_size=(7, self.embedding_dim),
                strides=1,
                activation=None,
                padding="valid"
            )
            self.conv1 = tf.squeeze(self.conv1, [2])
            self.conv1 = tf.nn.sigmoid(tf.nn.pool(self.conv1, window_shape=[2], pooling_type="MAX", strides=[2], padding="VALID"))

            self.conv2 = tf.layers.conv1d(
                inputs=self.conv1,
                filters=64,
                kernel_size=5,
                strides=1,
                activation=None,
                padding="valid"
            )
            self.conv2 = tf.nn.sigmoid(tf.nn.pool(self.conv2, window_shape=[2], pooling_type="MAX", strides=[2], padding="VALID"))

            self.conv3 = tf.layers.conv1d(
                inputs=self.conv2,
                filters=256,
                kernel_size=3,
                strides=1,
                activation=None,
                padding="valid"
            )
            self.conv3 = tf.nn.sigmoid(tf.nn.pool(self.conv3, window_shape=[2], pooling_type="MAX", strides=[2], padding="VALID"))

            self.conv4 = tf.layers.conv1d(
                inputs=self.conv3,
                filters=16,
                kernel_size=1,
                strides=1,
                activation=None,
                padding="valid"
            )
            self.conv4 = tf.nn.sigmoid(self.conv4)
            self.conv4 = tf.reshape(self.conv4, [-1, 13 * 16])

        self.cnn_out = self.conv4

        # lstm
        with tf.variable_scope("lstm_part" + self.scope):
            self.lstm = tf.nn.rnn_cell.LSTMCell(
                num_units=self.embedding_dim,
                use_peepholes=True,
                initializer=tf.contrib.layers.xavier_initializer(),
                num_proj=256,
                name="lstm_cell"
            )
            self.lstm_input = tf.squeeze(self.input_ph, [3])
            self.lstm_out, final_state = tf.nn.dynamic_rnn(cell=self.lstm, inputs=self.lstm_input, dtype=tf.float32
                                                     )
            self.lstm_out = tf.reduce_mean(self.lstm_out, keepdims=False, axis=1)

        self.fc_input = tf.concat([self.cnn_out, self.lstm_out], axis=-1)

        # fc
        with tf.variable_scope("fc_part" + self.scope):
            fc_out = self.fc_input
            for output_num in DNN_Shape:
                if output_num != 1:
                    fn = tf.nn.sigmoid
                else:
                    fn = None

                layer = tf.layers.dense(
                    inputs=fc_out,
                    units=output_num,
                    activation=fn
                )
                fc_out = layer

        self.nn_output = fc_out

        # about trainer and loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.nn_output, labels=self.target))
        self.trainer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def train(self, data):

        self.training_step = self.training_step + 1

        _, loss = self.sess.run([self.trainer, self.loss], feed_dict={
            self.input_ph: data['words'],
            self.target: data['tags']
        })

        return loss
        # if self.training_step % self.every_steps_save == 1:
        #     self.model_save()

    def get_result(self, data):

        output = self.sess.run([self.nn_output], feed_dict={
            self.input_ph: data['words'],
            self.target: data['tags']
        })
        return output

    def model_save(self, name=None):

        print("now training step %d...model saving..." % (self.training_step))
        if name == None:
            self.saver.save(self.sess, "model/training_step" + self.scope,
                            global_step=self.training_step)
        else:
            self.saver.save(self.sess, name)

    def model_load(self):

        self.saver.restore(self.sess, self.model_load_path)
