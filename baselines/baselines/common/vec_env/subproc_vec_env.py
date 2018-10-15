import numpy as np
from multiprocessing import Process, Pipe
from . import VecEnv, CloudpickleWrapper
from baselines.common.tile_images import tile_images
import time

USE_IMMITATION_ENV = True
if USE_IMMITATION_ENV:
    from TDCFeaturizer import TDCFeaturizer
    from train_featurizer import generate_dataset

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        if USE_IMMITATION_ENV:
            self.featurizer = TDCFeaturizer(92, 92, 84, 84, feature_vector_size=1024, learning_rate=0, experiment_name='default', is_variational=False)
            self.featurizer.load()
            video_dataset = generate_dataset('default', framerate=30/15, width=84, height=84)[0]
            self.featurized_dataset = self.featurizer.featurize(video_dataset)
            self.checkpoint_indexes = [0] * nenvs

            self.rewards = 0
            self.counter = 0

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs, rews, dones = np.stack(obs), np.stack(rews), np.stack(dones)
        if USE_IMMITATION_ENV:
            state_feature_vectors = self.featurizer.featurize(obs)
            dot_products = [np.dot(state_feature_vectors[i], self.featurized_dataset[self.checkpoint_indexes[i]]) for i in range(self.nenvs)]
            gamma_threshold = 0.5
            immitation_rewards = [0.5 if dot_product > gamma_threshold else 0 for dot_product in dot_products]

            rews += immitation_rewards
            mean_rews = np.mean(rews)
            self.rewards += mean_rews
            self.counter += 1
            if self.counter == 10000:
                print('10000 rewards: ', self.rewards)
                self.rewards = 0
                self.counter = 0
            #if dot_products[0] > 0.5:
            #    print(dot_products, immitation_rewards[0], rews)
            self.checkpoint_indexes = [self.checkpoint_indexes[i] + 1 if immitation_rewards[i] > 0 else self.checkpoint_indexes[i] for i in range(self.nenvs)]
            #print(self.checkpoint_indexes[0])

            self.checkpoint_indexes = [0 if dones[i] else self.checkpoint_indexes[i] for i in range(self.nenvs)]


        return obs, rews, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        self.checkpoint_indexes = [0] * self.nenvs
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        if self.viewer is not None:
            self.viewer.close()
        self.closed = True

    def render(self, mode='human'):
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(bigimg[:, :, ::-1])

        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError
