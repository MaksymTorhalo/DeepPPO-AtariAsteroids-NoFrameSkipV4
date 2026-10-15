import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward, RecordEpisodeStatistics
import ale_py
import numpy as np
import hparams




def make_env(seed, idx):
    def thunk():
        env = gym.make("ALE/Asteroids-v5",
                       frameskip=1,
                       repeat_action_probability=0.0,
                       obs_type='rgb')

        env = AtariPreprocessing(env,
                                 noop_max=30,
                                 frame_skip=4,
                                 screen_size=84,
                                 terminal_on_life_loss=True,
                                 grayscale_obs=True,
                                 scale_obs=False)

        env = RecordEpisodeStatistics(env)

        env = TransformReward(env, lambda r: np.sign(r))
        env = FrameStackObservation(env, 4)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
    return thunk