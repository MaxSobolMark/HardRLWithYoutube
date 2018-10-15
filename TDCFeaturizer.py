from BaseFeaturizer import BaseFeaturizer
import tensorflow as tf
import numpy as np
import datetime
import time
from residual_block import residual_block
import random

class TDCFeaturizer(BaseFeaturizer):
    """Temporal Distance Classification featurizer

    Reference: "Playing hard exploration games by watching YouTube"
    The unsupervised task consists of presenting the network with 2 frames separated by n timesteps,
    and making it classify the distance between the frames.

    We use the same network architecture as the paper:
        3 convolutional layers, followed by 3 residual blocks,
        followed by 2 fully connected layers for the encoder.

        For the classifier, we do a multiplication between both feature vectors
        followed by a fully connected layer.

    """

    def __init__(self, initial_width, initial_height, desired_width, desired_height, feature_vector_size=1024, learning_rate=0.0001, experiment_name='default'):
        print("Starting featurizer initialization")
        self.sess = tf.Session()
        self.graph = TDCFeaturizer._generate_featurizer(initial_width, initial_height, desired_width, desired_height, feature_vector_size, learning_rate)
        self.saver = tf.train.Saver()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.summary_writer = tf.summary.FileWriter('./summaries/{}/{}/'.format(experiment_name, timestamp), tf.get_default_graph())
        self.summary_writer.flush()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


    def _generate_featurizer(initial_width, initial_height, desired_width, desired_height, feature_vector_size, learning_rate):
        """Builds the TensorFlow graph for the featurizer

        Args:
            initial_width: The images' width before cropping.
            initial_height: The images' height before cropping.
            desired_width: Target width after cropping.
            desired_height: Target height after cropping.
            feature_fector_size: Length of the feature vector.
            learning_rate: Step size for the learning algorithm.

        Returns:
            graph object that contains the public nodes of the model.

        """
        is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")
        stacked_state = tf.placeholder(dtype=tf.float32, shape=(None, 2, initial_width, initial_height, 1), name="stacked_state")
        labels = tf.placeholder(dtype=tf.float32, shape=(None, 6), name="labels") # There are 6 possible labels

        state = tf.reshape(stacked_state, (-1, initial_width, initial_height, 1))
        state = tf.random_crop(state , (tf.shape(state)[0], desired_width, desired_height, 1))

        with tf.variable_scope('Encoder'):
            x = state
            for filters, strides in [(32, 2), (64, 1), (64, 1)]:
                x = TDCFeaturizer._convolutional_layer(x, filters, strides, is_training)
            for i in range(3):
                x = residual_block(x, 64, 64, 1, is_training)

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
                name="fv"
            )

            feature_vector = tf.nn.l2_normalize(feature_vector, -1)

            feature_vector_stack = tf.reshape(feature_vector, (-1, 2, feature_vector_size))

        with tf.variable_scope('Classifier'):
            combined_embeddings = tf.multiply(feature_vector_stack[:, 0, :], feature_vector_stack[:, 1, :])

            x = tf.layers.dense(
                combined_embeddings,
                feature_vector_size,
                #6,
                tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="fc1"
            )

            prediction = tf.layers.dense(
                x,
                6,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="prediction"
            )

        # Losses and optimizers
        with tf.variable_scope('losses', reuse=False):
            classifier_loss = tf.losses.softmax_cross_entropy(labels, prediction)
            accuracy, accuracy_update_op = tf.metrics.accuracy(
                labels=tf.argmax(labels, -1),
                predictions=tf.argmax(prediction, -1)
            )

            # We need to add the batch normalization update ops as dependencies
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(classifier_loss)


        classifier_loss_summary = tf.summary.scalar('classifier_loss', classifier_loss)
        accuracy_summary = tf.summary.scalar('accuracy', accuracy_update_op)
        summaries = tf.summary.merge([classifier_loss_summary, accuracy_summary])

        graph = {
            'is_training': is_training,
            'state': state,
            'labels': labels,
            'stacked_state': stacked_state,
            'feature_vector': feature_vector,
            'prediction': prediction,
            'loss': classifier_loss,
            'train_op': train_op,
            'summaries': summaries,
            'accuracy_update_op': accuracy_update_op
        }

        return graph

    def _convolutional_layer(input, filters, strides, is_training):
        """Constructs a conv2d layer followed by batch normalization, and max pooling"""
        x = tf.layers.conv2d(
            input,
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )

        x = tf.layers.batch_normalization(x, training=is_training)

        output = tf.layers.max_pooling2d(x, 2, 2)

        return output


    def train(self, dataset, epochs, batch_size):
        """Runs the training algorithm"""
        print("Starting training procedure")
        for epoch in range(epochs):
            frames, labels = TDCFeaturizer._generate_training_data(dataset, batch_size)
            feed_dict = {
                self.graph['is_training']: True,
                self.graph['stacked_state']: frames,
                self.graph['labels']: labels
            }
            run_ops = [
                self.graph['train_op'],
                self.graph['summaries'],
                self.graph['accuracy_update_op']
            ]
            _, train_summary, accuracy = self.sess.run(run_ops, feed_dict)
            if epoch % max(epochs // 100, 1) == 1:
                print('Epoch: {}/{}'.format(epoch, epochs))
                self.summary_writer.add_summary(train_summary, epoch)
        return True

    def _generate_training_data(videos, number_of_samples):
        """Constructs the unsupervised task
        
        Input data: two frames that are a number of timesteps apart
        Output: classify how many frames there are between the frames.
        """
        frames = np.empty((number_of_samples, 2, *videos[0].shape[1:]))
        labels = np.zeros((number_of_samples, 6))

        for i in range(number_of_samples):
            video_index = random.randint(0, len(videos) - 1)
            interval = random.randint(0, 5)
            if interval == 0:
                possible_frames_start = 0
                possible_frames_end = 0
            elif interval == 1:
                possible_frames_start = 1
                possible_frames_end = 1
            elif interval == 2:
                possible_frames_start = 2
                possible_frames_end = 2
            elif interval == 3:
                possible_frames_start = 3
                possible_frames_end = 4
            elif interval == 4:
                possible_frames_start = 5
                possible_frames_end = 20
            elif interval == 5:
                possible_frames_start = 21
                possible_frames_end = 200

            first_frame_index = random.randint(0, videos[video_index].shape[0] - possible_frames_end - 1)
            second_frame_index = random.randint(possible_frames_start, possible_frames_end)

            frames[i, 0] = videos[video_index][first_frame_index] / 255 # To normalize data
            frames[i, 1] = videos[video_index][first_frame_index + second_frame_index] / 255
            labels[i, interval] = 1.

        return frames, labels