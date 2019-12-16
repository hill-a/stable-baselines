import subprocess

import gym
import numpy as np

from stable_baselines.common.running_mean_std import RunningMeanStd
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from .test_common import _assert_eq

ENV_ID = 'Pendulum-v0'


def make_env():
    return gym.make(ENV_ID)


def test_runningmeanstd():
    """Test RunningMeanStd object"""
    for (x_1, x_2, x_3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2))]:
        rms = RunningMeanStd(epsilon=0.0, shape=x_1.shape[1:])

        x_cat = np.concatenate([x_1, x_2, x_3], axis=0)
        moments_1 = [x_cat.mean(axis=0), x_cat.var(axis=0)]
        rms.update(x_1)
        rms.update(x_2)
        rms.update(x_3)
        moments_2 = [rms.mean, rms.var]

        assert np.allclose(moments_1, moments_2)


def check_rms_equal(rmsa, rmsb):
    assert np.all(rmsa.mean == rmsb.mean)
    assert np.all(rmsa.var == rmsb.var)
    assert np.all(rmsa.count == rmsb.count)


def check_vec_norm_equal(norma, normb):
    assert norma.observation_space == normb.observation_space
    assert norma.action_space == normb.action_space
    assert norma.num_envs == normb.num_envs

    check_rms_equal(norma.obs_rms, normb.obs_rms)
    check_rms_equal(norma.ret_rms, normb.ret_rms)
    assert norma.clip_obs == normb.clip_obs
    assert norma.clip_reward == normb.clip_reward
    assert norma.norm_obs == normb.norm_obs
    assert norma.norm_reward == normb.norm_reward

    assert np.all(norma.ret == normb.ret)
    assert norma.gamma == normb.gamma
    assert norma.epsilon == normb.epsilon
    assert norma.training == normb.training


def test_vec_env(tmpdir):
    """Test VecNormalize Object"""
    clip_obs = 0.5
    clip_reward = 5.0

    orig_venv = DummyVecEnv([make_env])
    norm_venv = VecNormalize(orig_venv, norm_obs=True, norm_reward=True, clip_obs=clip_obs, clip_reward=clip_reward)
    _, done = norm_venv.reset(), [False]
    while not done[0]:
        actions = [norm_venv.action_space.sample()]
        obs, rew, done, _ = norm_venv.step(actions)
        assert np.max(np.abs(obs)) <= clip_obs
        assert np.max(np.abs(rew)) <= clip_reward

    path = str(tmpdir.join("vec_normalize"))
    norm_venv.save(path)
    deserialized = VecNormalize.load(path, venv=orig_venv)
    check_vec_norm_equal(norm_venv, deserialized)


def _make_warmstart_cartpole():
    """Warm-start VecNormalize by stepping through CartPole"""
    venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    venv = VecNormalize(venv)
    venv.reset()
    venv.get_original_obs()

    for _ in range(100):
        actions = [venv.action_space.sample()]
        venv.step(actions)
    return venv


def test_get_original():
    venv = _make_warmstart_cartpole()
    for _ in range(3):
        actions = [venv.action_space.sample()]
        obs, rewards, _, _ = venv.step(actions)
        obs = obs[0]
        orig_obs = venv.get_original_obs()[0]
        rewards = rewards[0]
        orig_rewards = venv.get_original_reward()[0]

        assert np.all(orig_rewards == 1)
        assert orig_obs.shape == obs.shape
        assert orig_rewards.dtype == rewards.dtype
        assert not np.array_equal(orig_obs, obs)
        assert not np.array_equal(orig_rewards, rewards)
        np.testing.assert_allclose(venv.normalize_obs(orig_obs), obs)
        np.testing.assert_allclose(venv.normalize_reward(orig_rewards), rewards)


def test_normalize_external():
    venv = _make_warmstart_cartpole()

    rewards = np.array([1, 1])
    norm_rewards = venv.normalize_reward(rewards)
    assert norm_rewards.shape == rewards.shape
    # Episode return is almost always >= 1 in CartPole. So reward should shrink.
    assert np.all(norm_rewards < 1)

    # Don't have any guarantees on obs normalization, except shape, really.
    obs = np.array([0, 0, 0, 0])
    norm_obs = venv.normalize_obs(obs)
    assert obs.shape == norm_obs.shape


def test_mpi_runningmeanstd():
    """Test RunningMeanStd object for MPI"""
    return_code = subprocess.call(['mpirun', '--allow-run-as-root', '-np', '2',
                                   'python', '-m', 'stable_baselines.common.mpi_running_mean_std'])
    _assert_eq(return_code, 0)


def test_mpi_moments():
    """
    test running mean std function
    """
    subprocess.check_call(['mpirun', '--allow-run-as-root', '-np', '3', 'python', '-c',
                           'from stable_baselines.common.mpi_moments '
                           'import _helper_runningmeanstd; _helper_runningmeanstd()'])
