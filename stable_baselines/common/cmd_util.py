"""
Helpers for scripts like run_atari.py.
"""

import os
import warnings

import gym

from stable_baselines import logger
from stable_baselines.bench import Monitor
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.misc_util import mpi_rank_or_zero
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_vec_env(env_id, n_envs=1, seed=None, start_index=0,
                 monitor_dir=None, wrapper_class=None,
                 env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id)
                if len(env_kwargs) > 0:
                    warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None,
                   start_index=0, allow_early_resets=True,
                   start_method=None, use_subprocess=False):
    """
    Create a wrapped, monitored VecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)

    # When using one environment, no need to start subprocesses
    if num_env == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)],
                         start_method=start_method)


def make_mujoco_env(env_id, seed, allow_early_resets=True):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The mujoco environment
    """
    set_global_seeds(seed + 10000 * mpi_rank_or_zero())
    env = gym.make(env_id)
    env = Monitor(env, os.path.join(logger.get_dir(), '0'), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env


def make_robotics_env(env_id, seed, rank=0, allow_early_resets=True):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) the rank of the environment (for logging)
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The robotic environment
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    keys = ['observation', 'desired_goal']
    # TODO: remove try-except once most users are running modern Gym
    try:  # for modern Gym (>=0.15.4)
        from gym.wrappers import FilterObservation, FlattenObservation
        env = FlattenObservation(FilterObservation(env, keys))
    except ImportError:  # for older gym (<=0.15.3)
        from gym.wrappers import FlattenDictWrapper  # pytype:disable=import-error
        env = FlattenDictWrapper(env, keys)
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',), allow_early_resets=allow_early_resets)
    env.seed(seed)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.

    :return: (ArgumentParser)
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.

    :return: (ArgumentParser) parser {'--env': 'BreakoutNoFrameskip-v4', '--seed': 0, '--num-timesteps': int(1e7)}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    return parser


def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.

    :return:  (ArgumentParser) parser {'--env': 'Reacher-v2', '--seed': 0, '--num-timesteps': int(1e6), '--play': False}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    return parser


def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.

    :return: (ArgumentParser) parser {'--env': 'FetchReach-v0', '--seed': 0, '--num-timesteps': int(1e6)}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser
