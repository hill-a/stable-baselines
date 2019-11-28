import os
import shutil

import pytest
import gym

from stable_baselines import A2C
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


@pytest.mark.parametrize("env_id", ['CartPole-v1', lambda: gym.make('CartPole-v1')])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("vec_env_cls", [None, SubprocVecEnv])
@pytest.mark.parametrize("wrapper_class", [None, gym.wrappers.TimeLimit])
def test_make_vec_env(env_id, n_envs, vec_env_cls, wrapper_class):
    env = make_vec_env(env_id, n_envs, vec_env_cls=vec_env_cls,
                       wrapper_class=wrapper_class, monitor_dir=None, seed=0)

    assert env.num_envs == n_envs

    if vec_env_cls is None:
        assert isinstance(env, DummyVecEnv)
        if wrapper_class is not None:
            assert isinstance(env.envs[0], wrapper_class)
        else:
            assert isinstance(env.envs[0], Monitor)
    else:
        assert isinstance(env, SubprocVecEnv)
    # Kill subprocesses
    env.close()


def test_custom_vec_env():
    """
    Stand alone test for a special case (passing a custom VecEnv class) to avoid doubling the number of tests.
    """
    monitor_dir = 'logs/test_make_vec_env/'
    env = make_vec_env('CartPole-v1', n_envs=1,
                       monitor_dir=monitor_dir, seed=0,
                       vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': None})


    assert env.num_envs == 1
    assert isinstance(env, SubprocVecEnv)
    assert os.path.isdir('logs/test_make_vec_env/')
    # Kill subprocess
    env.close()
    # Cleanup folder
    shutil.rmtree(monitor_dir)

    # This should fail because DummyVecEnv does not have any keyword argument
    with pytest.raises(TypeError):
        make_vec_env('CartPole-v1', n_envs=1, vec_env_kwargs={'dummy': False})


def test_evaluate_policy():
    model = A2C('MlpPolicy', 'Pendulum-v0', seed=0)
    n_steps_per_episode, n_eval_episodes = 200, 2
    model.n_callback_calls = 0

    def dummy_callback(locals_, _globals):
        locals_['model'].n_callback_calls += 1

    _, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes, deterministic=True,
                                 render=False, callback=dummy_callback, reward_threshold=None,
                                 return_episode_rewards=False)
    assert n_steps == n_steps_per_episode * n_eval_episodes
    assert n_steps == model.n_callback_calls

    # Reaching a mean reward of zero is impossible with the Pendulum env
    with pytest.raises(AssertionError):
        evaluate_policy(model, model.get_env(), n_eval_episodes, reward_threshold=0.0)

    episode_rewards, _ = evaluate_policy(model, model.get_env(), n_eval_episodes, return_episode_rewards=True)
    assert len(episode_rewards) == n_eval_episodes
