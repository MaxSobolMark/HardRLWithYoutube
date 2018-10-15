from BaseFeaturizer import BaseFeaturizer
import tensorflow as tf
import numpy as np
import datetime
import time
from residual_block import residual_block

class VAEFeaturizer(BaseFeaturizer):
    # Variational Auto Encoder featurizer

    def __init__(self, initial_width, initial_height, desired_width, desired_height, feature_vector_size=512, learning_rate=0.0001, experiment_name='default'):
        print("Starting featurizer initialization")
        self.sess = tf.Session()
        self.graph = VAEFeaturizer._generate_featurizer(initial_width, initial_height, feature_vector_size, learning_rate)
        self.saver = tf.train.Saver()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.summary_writer = tf.summary.FileWriter('./summaries/{}/{}/'.format(experiment_name, timestamp), tf.get_default_graph())
        self.summary_writer.flush()

        print("About to initialize vars")
        self.sess.run(tf.global_variables_initializer())


    def _generate_featurizer(width, height, feature_vector_size, learning_rate):
        # VAE based in https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
        is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")
        state = tf.placeholder(dtype=tf.float32, shape=(None, width, height, 1), name="state")

        # Encoder
        with tf.variable_scope('Encoder', reuse=False):
            # Uses the L3 Net architecture from "Look, Listen and Learn"
            #with tf.variable_scope('Block_1'):
                #block_1 = VAEFeaturizer._L3_net_layer(state, is_training, 32, (2, 2))
            #with tf.variable_scope('Block_2'):
                #block_2 = VAEFeaturizer._L3_net_layer(block_1, is_training, 64, (4, 4))

            # Uses ResNeXt architecture
            x = state
            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, 16, 16, 16, is_training, project_shortcut=project_shortcut)

            x = tf.layers.max_pooling2d(x, 2, 2)
            for i in range(3):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 16, 32, 16, is_training, strides=strides)

            x = tf.layers.max_pooling2d(x, 2, 2)
            for i in range(3):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 32, 64, 16, is_training, strides=strides)

            x = tf.layers.max_pooling2d(x, 2, 2)
            for i in range(3):
                strides = (2, 2) if i == 0 else (1, 1)
                x = residual_block(x, 64, 128, 16, is_training, strides=strides)

            x = tf.layers.max_pooling2d(x, 2, 2)

            flattened_features = tf.layers.flatten(x)

            with tf.variable_scope('FF'):
                fc1 = tf.layers.dense(
                    flattened_features,
                    feature_vector_size,
                    tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="fc1"
                )
                feature_vector_mean = tf.layers.dense(
                    fc1,
                    feature_vector_size,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="feature_vector_mean"
                )
                feature_vector_log_std = tf.layers.dense(
                    fc1,
                    feature_vector_size,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="feature_vector_log_std"
                )

            normal_dist = tf.random_normal(shape=tf.shape(feature_vector_log_std), mean=0, stddev=1, dtype=tf.float32)
            feature_vector = feature_vector_mean + tf.exp(feature_vector_log_std) * normal_dist

        # Decoder
        with tf.variable_scope('Decoder', reuse=False):

            decoder_input = tf.reshape(feature_vector, (-1, 1, 1, feature_vector_size))

            x = decoder_input

            for i in range(3):
                x = residual_block(x, feature_vector_size, feature_vector_size, 16, is_training)
            x = tf.layers.conv2d_transpose(
                x,
                filters=feature_vector_size,
                kernel_size=(3, 3),
                strides=2,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            #x = tf.layers.batch_normalization(x, training=is_training)

            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, feature_vector_size, feature_vector_size//2, 16, is_training, project_shortcut=project_shortcut)
            x = VAEFeaturizer._deconvolutional_layer(x, is_training, feature_vector_size//2)
            
            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, feature_vector_size//2, feature_vector_size//4, 16, is_training, project_shortcut=project_shortcut)
            x = VAEFeaturizer._deconvolutional_layer(x, is_training, feature_vector_size//4)

            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, feature_vector_size//4, feature_vector_size//8, 16, is_training, project_shortcut=project_shortcut)
            x = VAEFeaturizer._deconvolutional_layer(x, is_training, feature_vector_size//16)

            for i in range(3):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, feature_vector_size//16, 1, 16, is_training, project_shortcut=project_shortcut)
            decoder_output = tf.reshape(x, (-1, width, height, 1))

        # Losses and optimizers
        with tf.variable_scope('losses', reuse=False):
            reconstruction_loss = tf.squared_difference(state, decoder_output, name='reconstruction_loss')
            reconstruction_loss_mean = tf.reduce_mean(reconstruction_loss)

            kl_loss = -0.5 * tf.reduce_sum(
                1 + feature_vector_log_std - tf.square(feature_vector_mean) - tf.exp(feature_vector_log_std),
                axis=-1
            )
            kl_loss_mean = tf.reduce_mean(kl_loss)

            total_loss = reconstruction_loss + kl_loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)


        reconstruction_loss_summary = tf.summary.scalar('Reconstruction_loss', reconstruction_loss_mean)
        kl_loss_summary = tf.summary.scalar('KL_loss', kl_loss_mean)
        summaries = tf.summary.merge([reconstruction_loss_summary, kl_loss_summary])

        graph = {
            'is_training': is_training,
            'state': state,
            'feature_vector': feature_vector,
            'decoder_output': decoder_output,
            'reconstruction_loss_mean': reconstruction_loss_mean,
            'kl_loss_mean': kl_loss_mean,
            'train_op': train_op,
            'summaries': summaries
        }

        return graph



    def _L3_net_layer(input, is_training, filters, pooling_size):
        # Block from "Look, Listen and Learn" architecture
        output = tf.layers.conv2d(
            input,
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )

        #output = tf.layers.batch_normalization(output, training=is_training)

        output = tf.layers.conv2d(
            output,
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )

        #output = tf.layers.batch_normalization(output, training=is_training)

        output = tf.layers.max_pooling2d(output, pooling_size, pooling_size)

        return output

    def _deconvolutional_layer(input, is_training, filters):
        # Implements transposed convolutional layers. Returns data with double the shape of input
        output = tf.layers.conv2d_transpose(
            input,
            filters=filters,
            kernel_size=(3, 3),
            strides=2,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )
        #output = tf.layers.batch_normalization(output, training=is_training)
        output = tf.layers.conv2d_transpose(
            output,
            filters=filters,
            kernel_size=(3, 3),
            strides=2,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )
        #output = tf.layers.batch_normalization(output, training=is_training)
        return output

    def train(self, dataset, epochs, batch_size):
        print("Starting training procedure")
        np.random.shuffle(dataset)
        for epoch in range(epochs):
            print('Epoch: {}/{}'.format(epoch, epochs))
            for batch_index in range(dataset.shape[0] // batch_size):
                batch = dataset[batch_index * batch_size : batch_index * batch_size + batch_size]
                feed_dict = {
                    self.graph['is_training']: True,
                    self.graph['state']: batch
                }
                start_time = time.time()
                _, train_summary = self.sess.run([self.graph['train_op'], self.graph['summaries']], feed_dict)
                end_time = time.time()
                if batch_index == 0:
                    print("Train on one batch: ", end_time - start_time, "s")
            self.summary_writer.add_summary(train_summary, epoch)
        return True

    def defeaturize(self, features):
        # Features: ndarray shape = (None, feature_vector_size)
        return self.sess.run(self.graph['decoder_output'], {self.graph['is_training']: False, self.graph['feature_vector']: features})

    def reconstruct(self, data):
        # From images to reconstructed images
        return self.sess.run(self.graph['decoder_output'], {self.graph['is_training']: False, self.graph['state']: data})