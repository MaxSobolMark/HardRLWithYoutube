from immitation_env.atari.montezuma_immitation_env import MontezumaImmitationEnv
from gym.envs.registration import register

register(
    id='MontezumaImmitationNoFrameskip-v4',
    entry_point='immitation_env.atari:MontezumaImmitationEnv'
)