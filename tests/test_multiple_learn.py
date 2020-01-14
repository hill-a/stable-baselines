import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO2
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv

# TODO: Fix multiple-learn on commented-out models (Issue #619).
MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    PPO2,

    # MPI-based models, which use traj_segment_generator instead of Runner.
    #
    # PPO1,
    # TRPO,

    # Off-policy models, which don't use Runner but reset every .learn() anyways.
    #
    # DDPG,
    # SAC,
    # TD3,
]


@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_model_multiple_learn_no_reset(model_class):
    """Check that when we call learn multiple times, we don't unnecessarily
    reset the environment.
    """
    if model_class is ACER:
        def make_env():
            return IdentityEnv(ep_length=1e10, dim=2)
    else:
        def make_env():
            return IdentityEnvBox(ep_length=1e10)
    env = make_env()
    venv = DummyVecEnv([lambda: env])
    model = model_class(policy="MlpPolicy", env=venv)
    _check_reset_count(model, env)

    # Try again following a `set_env`.
    env = make_env()
    venv = DummyVecEnv([lambda: env])
    assert env.num_resets == 0

    model.set_env(venv)
    _check_reset_count(model, env)


def _check_reset_count(model, env: IdentityEnv):
    assert env.num_resets == 0
    _prev_runner = None
    for _ in range(4):
        model.learn(total_timesteps=400)
        # Lazy constructor for Runner fires upon the first call to learn.
        assert env.num_resets == 1
        if _prev_runner is not None:
            assert _prev_runner is model.runner, "Runner shouldn't change"
        _prev_runner = model.runner
