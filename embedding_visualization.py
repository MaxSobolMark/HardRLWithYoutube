import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

def visualize_embeddings(embeddings, experiment_name='default'):
    """Save the embeddings to be visualised using t-sne on TensorBoard
    
    Based on https://medium.com/@vegi/visualizing-higher-dimensional-data-using-t-sne-on-tensorboard-7dbf22682cf2
    """
    tf_embeddings = tf.Variable(np.concatenate(embeddings, 0))

    # Generate metadata
    metadata = 'video_index\tframe_index\n'
    for video_index in range(len(embeddings)):
        for frame_index in range(embeddings[video_index].shape[0]):
            metadata += '{}\t{}\n'.format(video_index, frame_index)

    metadata_path = 'embeddings/{}/labels.tsv'.format(experiment_name)
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(metadata)


    with tf.Session() as sess:
        saver = tf.train.Saver([tf_embeddings])
        sess.run(tf_embeddings.initializer)
        saver.save(sess, 'embeddings/{}/embeddings.ckpt'.format(experiment_name))
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()

        embedding.tensor_name = tf_embeddings.name
        embedding.metadata_path = metadata_path.split('/')[-1]

        projector.visualize_embeddings(tf.summary.FileWriter('embeddings/{}'.format(experiment_name)), config)