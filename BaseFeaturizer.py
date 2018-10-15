from abc import ABCMeta, abstractmethod # Required to create an abstract class
import numpy as np
from scipy import spatial # For nearest neighbour lookup
import tensorflow as tf

class BaseFeaturizer(metaclass=ABCMeta):
    """Interface for featurizers

    Featurizers take images that are potentially from different videos of the same game,
    and encode them into an aligned embedding (the same frames from different videos
    have similar embeddings.)

    The mapping between images and embeddings is learned by training with an unsupervised task.
    Each featurizer type uses a different unsupervised task.

    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, dataset, epochs, batch_size):
        """Trains the featurizer.
        Dataset is a list of ndarrays; overall shape = (videos, frames, width, height, 1)
        """
        raise NotImplementedError

    def featurize(self, data, batch_size=32):
        """Encodes the data into an embedding
        Data: ndarray with shape (-1, width, height, 1)
        """
        splitted_data = np.array_split(data, max(data.shape[0] // batch_size, 1))
        feature_vectors = []
        for batch in splitted_data:
            normalized_batch = batch / 255
            feature_vectors.append(self.sess.run(self.graph['feature_vector'], {
                self.graph['is_training']: False, self.graph['state']: normalized_batch
            }))
        feature_vectors = np.concatenate(feature_vectors)
        return feature_vectors

    def save(self, save_path='default'):
        return self.saver.save(self.sess, './featurizers/{}/{}.ckpt'.format(save_path, save_path))

    def load(self, load_path='default'):
        self.saver.restore(self.sess, './featurizers/{}/{}.ckpt'.format(load_path, load_path))

    def evaluate_cycle_consistency(self, data, sequence_length=1024):
        '''Cycle-consistency evaluation as in "Playing hard exploration games by watching YouTube"'''
        shuffled_data = np.copy(data)
        np.random.shuffle(shuffled_data)
        first_sequence = shuffled_data[:sequence_length]
        second_sequence = shuffled_data[sequence_length : 2 * sequence_length]

        # featurize sequences
        first_sequence_features = self.featurize(first_sequence)
        second_sequence_features = self.featurize(second_sequence)
        
        # Use k-dimensional trees to do nearest-neighbor lookup
        first_sequence_kd_tree = spatial.KDTree(first_sequence_features)
        second_sequence_kd_tree = spatial.KDTree(second_sequence_features)

        consistent_cycles = 0
        for i in range(sequence_length):
            v = first_sequence_features[i]
            _, w_index = second_sequence_kd_tree.query(v)
            w = second_sequence_features[w_index]
            _, v_prime_index = first_sequence_kd_tree.query(w)
            if i == v_prime_index:
                consistent_cycles += 1

        return consistent_cycles / sequence_length