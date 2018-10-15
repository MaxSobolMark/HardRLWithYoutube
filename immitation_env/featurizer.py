from TDCFeaturizer import TDCFeaturizer
from train_featurizer import generate_dataset
import tensorflow as tf

def init():
    global featurizer
    global featurized_dataset
    featurizer = TDCFeaturizer(92, 92, 84, 84, feature_vector_size=1024, learning_rate=0, experiment_name='default', is_variational=False)
    print('test: ', featurizer.sess.run(tf.constant(1)+4))
    featurizer.load()
    print('sess: {}'.format(featurizer.sess))
    video_dataset = generate_dataset('default', framerate=30/15, width=84, height=84)[0]
    featurized_dataset = featurizer.featurize(video_dataset)