import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO1, PPO2, TRPO
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv

# TODO: Fix multiple-learn on commented-out models.
MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    # DDPG,
    PPO1,
    PPO2,
    # SAC,
    # TD3,
    TRPO
]


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_model_multiple_learn_no_reset(model_class):
    """Check that when we call learn multiple times, we don't unnecessarily
    reset the environment.
    """
    if model_class is ACER:
        def make_env(): return IdentityEnv(ep_length=1e10, dim=2)
    else:
        def make_env(): return IdentityEnvBox(ep_length=1e10)
    venv = DummyVecEnv([make_env])
    env = venv.envs[0]
    assert env.num_resets == 0

    model = model_class(policy="MlpPolicy", env=venv)
    for _ in range(4):
        model.learn(total_timesteps=10)
        assert env.num_resets == 1
