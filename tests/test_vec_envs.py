import gym
import numpy as np
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

ENV_ID = 'BreakoutNoFrameskip-v4'


def test_dummyvecenv():
    """test dummy_vec_env object"""

    test_vecenv(DummyVecEnv([lambda: gym.make(ENV_ID) for i in range(3)]))


def test_subprocvecenv():
    """test dummy_vec_env object"""

    test_vecenv(SubprocVecEnv([lambda: gym.make(ENV_ID) for i in range(3)]))


def test_vecenv(vec_env):
    """test arbitrary vectorized environment"""

    envmethod_results = vec_env.env_method('reset')
    setattr_results = []
    for i in range(3):
        setattr_results.append(vec_env.set_attr('name', 'testname' + str(i), indices=i))
    getattr_results = vec_env.get_attr('name')
    assert [res.shape for res in envmethod_results] == [(210, 160, 3)] * 3
    assert setattr_results == [[None]] * 3
    assert getattr_results == ['testname0', 'testname1', 'testname2']


if __name__ == '__main__':
    test_dummyvecenv()
    test_subprocvecenv()
    exit(0)
