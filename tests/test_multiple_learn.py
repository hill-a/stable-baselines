import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO2
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv

# TODO: Fix multiple-learn on commented-out models.
MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    PPO2,

    # ActorCriticRLModel, uses Runner-like interface that doesn't extend
    # `AbstractEnvRunner`. (More specifically, uses a "segment" generator from
    # TRPO's utils.)
    #
    # TODO: Modify `trpo.utils.traj_segment_generator` so that it follows
    # the `AbstractEnvRunner` interface, and then join the `ActorCriticRLModel`
    # and `ActorCriticRLModelWithRunner` classes.
    #
    # Why is this hard in the first place? Some sort of dependency loop, if I
    # remember correctly. In TRPO, something depends on the Runner which
    # depends on the ActorCriticRLModel's policy.
    #
    # Maybe I can make this happen today. That way the PR will be much easier
    # to explain I think.
    #
    # What about we do it this way. I'll make a PR with the old version (two
    # classes). Then I'll run my batch size experiment using this PR. Then
    # I will try to make a PR that joins the two `ActorCriticRLModel`s.
    #
    # PPO1,
    # TRPO,

    # Off-policy, doesn't use Runner.
    # TODO: Make these models pass `test_mdodel_multiple_learn_no_reset` as
    # well.
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
