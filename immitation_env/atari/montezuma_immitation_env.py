import gym
from .. import ImmitationWrapper
from TDCFeaturizer import TDCFeaturizer
from train_featurizer import generate_dataset

class MontezumaImmitationEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        featurizer = TDCFeaturizer(92, 92, 84, 84, feature_vector_size=1024, learning_rate=0, experiment_name='default', is_variational=False)
        featurizer.load()
        video_dataset = generate_dataset('default', framerate=30/15, width=84, height=84)[0]
        featurized_dataset = featurizer.featurize(video_dataset)
        self._env = ImmitationWrapper(gym.make('MontezumaRevengeNoFrameskip-v4'), featurizer=featurizer, featurized_dataset=featurized_dataset)
        self.observation_space = self._env.unwrapped.observation_space
        self.action_space = self._env.unwrapped.action_space
        self.np_random = self._env.unwrapped.np_random
        self.ale = self._env.unwrapped.ale

    def seed(self, seed = None):
        return self._env.seed(seed)

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        return self._env.unwrapped.render(mode)

    def close(self):
        self._env.unwrapped.close()

    def get_action_meanings(self):
        return self._env.unwrapped.get_action_meanings()