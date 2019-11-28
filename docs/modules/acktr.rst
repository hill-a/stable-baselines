.. _acktr:

.. automodule:: stable_baselines.acktr


ACKTR
=====

`Actor Critic using Kronecker-Factored Trust Region (ACKTR) <https://arxiv.org/abs/1708.05144>`_ uses
Kronecker-factored approximate curvature (K-FAC) for trust region optimization.


Notes
-----

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- ``python -m stable_baselines.acktr.run_atari`` runs the algorithm for 40M frames = 10M timesteps on an Atari game.
  See help (``-h``) for more options.

Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
  from stable_baselines.common import make_vec_env
  from stable_baselines import ACKTR

  # multiprocess environment
  env = make_vec_env('CartPole-v1', n_envs=4)

  model = ACKTR(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("acktr_cartpole")

  del model # remove to demonstrate saving and loading

  model = ACKTR.load("acktr_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: ACKTR
  :members:
  :inherited-members:
