.. _callbacks:

Callbacks
=========

A callback is a set of functions that will be called at given stages of the training procedure.
You can use callbacks to access internal state of the RL model during training.
It allows one to do monitoring, auto saving, model manipulation, progress bars, ...


Custom Callback
---------------

To build a custom callback, you need to create a class that derives from ``BaseCallback``.
This will give you access to events (``_on_training_start``, ``_on_step``) and useful variables (like `self.model` for the RL model).


You can find two examples of custom callbacks in the documentation: one for saving the best model according to the training reward (see :ref:`Examples <examples>`), and one for logging additional values with Tensorboard (see :ref:`Tensorboard section <tensorboard>`).


.. code-block:: python

    from stable_baselines.common.callbacks import BaseCallback


    class CustomCallback(BaseCallback):
        """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
        """
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            # Those variables will be accessible in the callback
            # (they are defined in the base class)
            # The RL model
            # self.model = None  # type: BaseRLModel
            # An alias for self.model.get_env(), the environment used for training
            # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
            # Number of time the callback was called
            # self.n_calls = 0  # type: int
            # self.num_timesteps = 0  # type: int
            # local and global variables
            # self.locals = None  # type: Dict[str, Any]
            # self.globals = None  # type: Dict[str, Any]
            # The logger object, used to report things in the terminal
            # self.logger = None  # type: logger.Logger
            # # Sometimes, for event callback, it is useful
            # # to have access to the parent object
            # self.parent = None  # type: Optional[BaseCallback]

        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            pass

        def _on_rollout_start(self) -> None:
            """
            A rollout is the collection of environment interaction
            using the current policy.
            This event is triggered before collecting new samples.
            """
            pass

        def _on_step(self) -> bool:
            """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """
            return True

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            pass

        def _on_training_end(self) -> None:
            """
            This event is triggered before exiting the `learn()` method.
            """
            pass


.. note::
  `self.num_timesteps` corresponds to the total number of steps taken in the environment, i.e., it is the number of environments multiplied by the number of time `env.step()` was called

  You should know that ``PPO1`` and ``TRPO`` update `self.num_timesteps` after each rollout (and not each step) because they rely on MPI.

  For the other algorithms, `self.num_timesteps` is incremented by ``n_envs`` (number of environments) after each call to `env.step()`


.. note::

  For off-policy algorithms like SAC, DDPG, TD3 or DQN, the notion of ``rollout`` corresponds to the steps taken in the environment between two updates.


.. _EventCallback:

Event Callback
--------------

Compared to Keras, Stable Baselines provides a second type of ``BaseCallback``, named ``EventCallback`` that is meant to trigger events. When an event is triggered, then a child callback is called.

As an example, :ref:`EvalCallback` is an ``EventCallback`` that will trigger its child callback when there is a new best model.
A child callback is for instance :ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>` that stops the training if the mean reward achieved by the RL model is above a threshold.

.. note::

	We recommend to take a look at the source code of :ref:`EvalCallback` and :ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>` to have a better overview of what can be achieved with this kind of callbacks.


.. code-block:: python

    class EventCallback(BaseCallback):
        """
        Base class for triggering callback on event.

        :param callback: (Optional[BaseCallback]) Callback that will be called
            when an event is triggered.
        :param verbose: (int)
        """
        def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
            super(EventCallback, self).__init__(verbose=verbose)
            self.callback = callback
            # Give access to the parent
            if callback is not None:
                self.callback.parent = self
        ...

        def _on_event(self) -> bool:
            if self.callback is not None:
                return self.callback()
            return True



Callback Collection
-------------------

Stable Baselines provides you with a set of common callbacks for:

- saving the model periodically (:ref:`CheckpointCallback`)
- evaluating the model periodically and saving the best one (:ref:`EvalCallback`)
- chaining callbacks (:ref:`CallbackList`)
- triggering callback on events (:ref:`EventCallback`, :ref:`EveryNTimesteps`)
- stopping the training early based on a reward threshold (:ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>`)


.. _CheckpointCallback:

CheckpointCallback
^^^^^^^^^^^^^^^^^^

Callback for saving a model every ``save_freq`` steps, you must specify a log folder (``save_path``)
and optionally a prefix for the checkpoints (``rl_model`` by default).


.. code-block:: python

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import CheckpointCallback
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix='rl_model')

    model = SAC('MlpPolicy', 'Pendulum-v0')
    model.learn(2000, callback=checkpoint_callback)


.. _EvalCallback:

EvalCallback
^^^^^^^^^^^^

Evaluate periodically the performance of an agent, using a separate test environment.
It will save the best model if ``best_model_save_path`` folder is specified and save the evaluations results in a numpy archive (`evaluations.npz`) if ``log_path`` folder is specified.


.. note::

	You can pass a child callback via the ``callback_on_new_best`` argument. It will be triggered each time there is a new best model.



.. code-block:: python

    import gym

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import EvalCallback

    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)

    model = SAC('MlpPolicy', 'Pendulum-v0')
    model.learn(5000, callback=eval_callback)


.. _Callbacklist:

CallbackList
^^^^^^^^^^^^

Class for chaining callbacks, they will be called sequentially.
Alternatively, you can pass directly a list of callbacks to the `learn()` method, it will be converted automatically to a ``CallbackList``.


.. code-block:: python

    import gym

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/')
    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=500)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    model = SAC('MlpPolicy', 'Pendulum-v0')
    # Equivalent to:
    # model.learn(5000, callback=[checkpoint_callback, eval_callback])
    model.learn(5000, callback=callback)


.. _StopTrainingCallback:

StopTrainingOnRewardThreshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stop the training once a threshold in episodic reward (mean episode reward over the evaluations) has been reached (i.e., when the model is good enough).
It must be used with the :ref:`EvalCallback` and use the event triggered by a new best model.


.. code-block:: python

    import gym

    from stable_baselines import SAC
    from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    # Separate evaluation env
    eval_env = gym.make('Pendulum-v0')
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

    model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
    # Almost infinite number of timesteps, but the training will stop
    # early as soon as the reward threshold is reached
    model.learn(int(1e10), callback=eval_callback)


.. _EveryNTimesteps:

EveryNTimesteps
^^^^^^^^^^^^^^^

An :ref:`EventCallback` that will trigger its child callback every ``n_steps`` timesteps.


.. note::

	Because of the way ``PPO1`` and ``TRPO`` work (they rely on MPI), ``n_steps`` is a lower bound between two events.


.. code-block:: python

  import gym

  from stable_baselines import PPO2
  from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps

  # this is equivalent to defining CheckpointCallback(save_freq=500)
  # checkpoint_callback will be triggered every 500 steps
  checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/')
  event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

  model = PPO2('MlpPolicy', 'Pendulum-v0', verbose=1)

  model.learn(int(2e4), callback=event_callback)


.. automodule:: stable_baselines.common.callbacks
  :members:


  Legacy: A functional approach
  -----------------------------

  .. warning::

  	This way of doing callbacks is deprecated in favor of the object oriented approach.



  A callback function takes the ``locals()`` variables and the ``globals()`` variables from the model, then returns a boolean value for whether or not the training should continue.

  Thanks to the access to the models variables, in particular ``_locals["self"]``, we are able to even change the parameters of the model without halting the training, or changing the model's code.


  .. code-block:: python

      from typing import Dict, Any

      from stable_baselines import PPO2


      def simple_callback(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> bool:
          """
          Callback called at each step (for DQN and others) or after n steps (see ACER or PPO2).
          This callback will save the model and stop the training after the first call.

          :param _locals: (Dict[str, Any])
          :param _globals: (Dict[str, Any])
          :return: (bool) If your callback returns False, training is aborted early.
          """
          print("callback called")
          # Save the model
          _locals["self"].save("saved_model")
          # If you want to continue training, the callback must return True.
          # return True # returns True, training continues.
          print("stop training")
          return False # returns False, training stops.

      model = PPO2('MlpPolicy', 'CartPole-v1')
      model.learn(2000, callback=simple_callback)
