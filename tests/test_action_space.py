import pytest
import numpy as np

from stable_baselines import A2C, PPO1, PPO2, TRPO
from stable_baselines.common.identity_env import IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

MODEL_LIST = [
    A2C,
    PPO1,
    PPO2,
    TRPO
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multidiscrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiDiscrete(10)])

    model = model_class("MlpPolicy", env)
    model.learn(total_timesteps=1000)
    evaluate_policy(model, env, n_eval_episodes=5)
    obs = env.reset()

    assert np.array(model.action_probability(obs)).shape == (2, 1, 10), \
        "Error: action_probability not returning correct shape"
    assert np.prod(model.action_probability(obs, actions=env.action_space.sample()).shape) == 1, \
        "Error: not scalar probability"


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_identity_multibinary(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multibinary action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: IdentityEnvMultiBinary(10)])

    model = model_class("MlpPolicy", env)
    model.learn(total_timesteps=1000)
    evaluate_policy(model, env, n_eval_episodes=5)
    obs = env.reset()

    assert model.action_probability(obs).shape == (1, 10), \
        "Error: action_probability not returning correct shape"
    assert np.prod(model.action_probability(obs, actions=env.action_space.sample()).shape) == 1, \
        "Error: not scalar probability"
