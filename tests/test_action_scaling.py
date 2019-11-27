import pytest
import numpy as np

from stable_baselines import DDPG, TD3, SAC
from stable_baselines.common.identity_env import IdentityEnvBox

ROLLOUT_STEPS = 100

MODEL_LIST = [
    (DDPG, dict(nb_train_steps=0, nb_rollout_steps=ROLLOUT_STEPS)),
    (TD3, dict(train_freq=ROLLOUT_STEPS + 1, learning_starts=0)),
    (SAC, dict(train_freq=ROLLOUT_STEPS + 1, learning_starts=0)),
    (TD3, dict(train_freq=ROLLOUT_STEPS + 1, learning_starts=ROLLOUT_STEPS)),
    (SAC, dict(train_freq=ROLLOUT_STEPS + 1, learning_starts=ROLLOUT_STEPS))
]


@pytest.mark.parametrize("model_class, model_kwargs", MODEL_LIST)
def test_buffer_actions_scaling(model_class, model_kwargs):
    """
    Test if actions are scaled to tanh co-domain before being put in a buffer
    for algorithms that use tanh-squashing, i.e., DDPG, TD3, SAC

    :param model_class: (BaseRLModel) A RL Model
    :param model_kwargs: (dict) Dictionary containing named arguments to the given algorithm
    """

    # check random and inferred actions as they possibly have different flows
    for random_coeff in [0.0, 1.0]:

        env = IdentityEnvBox(-2000, 1000)

        model = model_class("MlpPolicy", env, seed=1, random_exploration=random_coeff, **model_kwargs)
        model.learn(total_timesteps=ROLLOUT_STEPS)

        assert hasattr(model, 'replay_buffer')

        buffer = model.replay_buffer

        assert buffer.can_sample(ROLLOUT_STEPS)

        _, actions, _, _, _ = buffer.sample(ROLLOUT_STEPS)

        assert not np.any(actions > np.ones_like(actions))
        assert not np.any(actions < -np.ones_like(actions))
