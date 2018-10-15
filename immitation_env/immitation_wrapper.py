import gym
import numpy as np
from train_featurizer import preprocess_image

class ImmitationWrapper(gym.Wrapper):
    def __init__(self, env, featurizer, featurized_dataset):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.featurizer = featurizer
        self.featurized_dataset = featurized_dataset
        self.checkpoint_index = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state_feature_vector = self.featurizer.featurize(np.expand_dims(preprocess_image(obs, 84, 84), 0))
        immitation_reward = 0
        gamma_threshold = 0.5
        if np.dot(state_feature_vector, self.featurized_dataset[self.checkpoint_index]) > gamma_threshold:
            immitation_reward = 0.5
            self.checkpoint_index += 1

        return obs, reward + immitation_reward, done, info

    def reset(self, **kwargs):
        self.checkpoint_index = 0
        return self.env.reset(**kwargs)