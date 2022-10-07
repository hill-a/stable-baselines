import os

import numpy as np
import tensorflow as tf
import pytest
from gym import make
from gym.wrappers.time_limit import TimeLimit

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc


def custom_cnn_extractor(input_images):
    activ = tf.nn.relu
    layer_1 = activ(conv(input_images, 'c1', n_filters=8, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=8, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    layer_2 = conv_to_fc(layer_2)
    return activ(linear(layer_2, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))


class CustomCnnLstmPolicy1(CnnLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=32, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         cnn_extractor=custom_cnn_extractor, **_kwargs)


class CustomCnnLstmPolicy2(CnnLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=32, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=['lstm', 8], cnn_extractor=custom_cnn_extractor, **_kwargs)


class CustomCnnLstmPolicy3(CnnLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=32, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         cnn_extractor=custom_cnn_extractor, **_kwargs)


POLICIES = [CnnLstmPolicy, CustomCnnLstmPolicy1, CustomCnnLstmPolicy2, CustomCnnLstmPolicy3]


def make_env(i):
    env = make("Breakout-v0")
    env = TimeLimit(env, max_episode_steps=20)
    env.seed(i)
    return env


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.expensive
def test_cnn_lstm_policy(request, policy):
    model_fname = './test_model_{}.zip'.format(request.node.name)

    try:
        env = make_env(0)
        model = PPO2(policy, env, nminibatches=1)
        model.learn(total_timesteps=15)
        env = model.get_env()
        evaluate_policy(model, env, n_eval_episodes=5)
        # saving
        model.save(model_fname)
        del model, env
        # loading
        _ = PPO2.load(model_fname, policy=policy)

    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)
