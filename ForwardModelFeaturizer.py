from BaseFeaturizer import BaseFeaturizer
import tensorflow as tf
import numpy as np
from residual_block import residual_block
from TDCFeaturizer import TDCFeaturizer
import datetime
import time


class ForwardModelFeaturizer(BaseFeaturizer):
    """ Forward Model Featurizer

    Encodes images into an embedding, and minimizes the KL divergence between f(s(t)), and forwardModel(f(s(t+1)))

    """

    def __init__(self, initial_width, initial_height, desired_width, desired_height, feature_vector_size=16, learning_rate=0.0001, is_variational=False, experiment_name='default'):
        print("Starting featurizer initialization")
        self.sess = tf.Session()
        self.model = Model(initial_width, initial_height, desired_width, desired_height, feature_vector_size, learning_rate)
        self.saver = tf.train.Saver()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.summary_writer = tf.summary.FileWriter('./summaries/{}/{}/'.format(experiment_name, timestamp), tf.get_default_graph())
        self.summary_writer.flush()

        print("About to initialize vars")
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def featurize(self, data):
        # Data: ndarray shape = (None, width, height, 1)
        splitted_data = np.array_split(data, max(data.shape[0] // 32, 1))
        feature_vectors = []
        for batch in splitted_data:
            normalized_batch = batch / 255
            feature_vectors.append(self.sess.run(self.model.feature_vector, {
                self.model.is_training: False,
                self.model.state: normalized_batch,
                self.model.next_state: np.zeros([1, *normalized_batch.shape[1:]])
            }))
        feature_vectors = np.concatenate(feature_vectors)
        return feature_vectors

    def train(self, dataset, epochs, batch_size):
        print("Starting training procedure")
        for epoch in range(epochs):
            frames, next_frames = ForwardModelFeaturizer._generate_training_data(dataset, batch_size)
            feed_dict = {
                self.model.is_training: True,
                self.model.state: frames,
                self.model.next_state: next_frames
            }
            start_time = time.time()
            _, train_summary = self.sess.run([self.model.train_op, self.model.summaries], feed_dict)
            end_time = time.time()
            if epoch % max(epochs // 100, 1) == 1:
                print('Epoch: {}/{}'.format(epoch, epochs))
                self.summary_writer.add_summary(train_summary, epoch)
        return True

    def _generate_training_data(videos, number_of_samples):
        concatenated_videos = np.concatenate(videos, 0)
        frames_indexes = np.random.choice(len(concatenated_videos) - 2, number_of_samples, replace=False)
        next_frames_indexes = frames_indexes + 1

        frames = np.array([concatenated_videos[index] for index in frames_indexes]) / 255 # To normalize data
        next_frames = np.array([concatenated_videos[index] for index in next_frames_indexes]) / 255

        return frames, next_frames


class Model(object):

    def __init__(self, initial_width, initial_height, desired_width, desired_height, feature_vector_size, learning_rate):
        self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, initial_width, initial_height, 1), name="state")
        self.next_state = tf.placeholder(dtype=tf.float32, shape=(None, initial_width, initial_height, 1), name="next_state")

        combined_states = tf.concat([self.state, self.next_state], 0, name="concat_states")
        combined_states = tf.random_crop(combined_states, (tf.shape(combined_states)[0], desired_width, desired_height, 1))

        # Encoder
        with tf.variable_scope('Encoder'):
            x = combined_states
            for filters, strides in [(32, 2), (64, 1), (64, 1)]:
                x = TDCFeaturizer._convolutional_layer(x, filters, strides, self.is_training)
            for i in range(3):
                x = residual_block(x, 64, 64, 1, self.is_training)

            x = tf.layers.flatten(x)

            x = tf.layers.dense(
                x,
                feature_vector_size,
                tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="fc1"
            )

            # Compose feature vector

            feature_vector = tf.layers.dense(
                x,
                feature_vector_size,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="feature_vector"
            )

            feature_vector = tf.nn.l2_normalize(feature_vector, -1)

            self.feature_vector = feature_vector[:tf.shape(self.state)[0]]
            self.next_feature_vector = feature_vector[tf.shape(self.state)[0]:]

        # Forward Model
        with tf.variable_scope('forward_model'):
            x = self.feature_vector
            for i in range(3):
                x = tf.layers.dense(
                    x,
                    feature_vector_size,
                    tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="fc"+str(i)
                )

            operation_vector =  tf.layers.dense(
                x,
                feature_vector_size,
                tf.nn.sigmoid,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="operation_vector"
            )

            self.predicted_next_feature_vector = self.feature_vector + operation_vector

            self.predicted_next_feature_vector = tf.nn.l2_normalize(self.predicted_next_feature_vector, -1)

        # Losses and optimizers
        with tf.variable_scope('losses'):
            forward_model_loss = tf.losses.absolute_difference(self.next_feature_vector, self.predicted_next_feature_vector)
            total_loss = forward_model_loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

        forward_model_loss_summary = tf.summary.scalar('forward_model_loss', forward_model_loss)
        self.summaries = tf.summary.merge([forward_model_loss_summary])