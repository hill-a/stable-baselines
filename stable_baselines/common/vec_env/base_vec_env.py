from abc import ABC, abstractmethod
import pickle

import cloudpickle
from stable_baselines import logger


class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: (int) the number of environments
    :param observation_space: (Gym Space) the observation space
    :param action_space: (Gym Space) the action space
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: ([int] or [float]) observation
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environment's resources.
        """
        pass

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        pass

    @abstractmethod
    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        pass

    @abstractmethod
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        pass

    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    def render(self, *args, **kwargs):
        """
        Gym environment rendering

        :param mode: (str) the rendering type
        """
        logger.warn('Render not defined for %s' % self)

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name (str): name of attribute to check for
        :param already_found (bool): whether this attribute has already been found in a wrapper
        :return: (str or None) blocked_venv_class
        """
        if hasattr(self, name) and already_found:
            blocked_venv_class = "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            blocked_venv_class = None
        return blocked_venv_class


    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class

    :param venv: (VecEnv) the vectorized environment to wrap
    :param observation_space: (Gym Space) the observation space (can be None to load from venv)
    :param action_space: (Gym Space) the action space (can be None to load from venv)
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self, num_envs=venv.num_envs, observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, *args, **kwargs):
        return self.venv.render(*args, **kwargs)

    def get_images(self):
        return self.venv.get_images()

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def __getattr__(self, name):
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        return self.getattr_recursive(name, top_level=True)

    def getattr_recursive(self, name, top_level):
        """Recursively check wrappers to find attribute. Also ensure that there is no
        ambiguity about which wrapper's attribute is being referenced.

        :param name (str) name of attribute to look for
        :param top_level (bool) whether this was the first call to this recursive function.
               Used to avoid unnecessary calls to self.getattr_depth_check.
        :return: (object) attr
        """
        if hasattr(self.venv, 'getattr_recursive'):
            attr = self.venv.getattr_recursive(name, top_level=False)
        else:
            attr = getattr(self.venv, name)  # get from an unwrapped venv

        if top_level:
            blocked_class = self.getattr_depth_check(name, already_found=False)
            if blocked_class is not None:
                own_class = "{0}.{1}".format(type(self).__module__, type(self).__name__)
                msg = """Error: Recursive attribute lookup for {0} from {1} is
                ambiguous and hides attribute from {2}""".format(name, own_class, blocked_class)
                raise AttributeError(msg)

        return attr

    def getattr_depth_check(self, name, already_found):
        """See base class.

        :return: (str or None) blocked_wrapper_class
        """
        if hasattr(self, name) and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            blocked_wrapped_class = "{0}.{1}".format(type(self).__module__, type(self).__name__)
        elif hasattr(self, name) and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            blocked_wrapper_class = self.venv.getattr_depth_check(True, name)
        else:
            # this wrapper does not have the attribute. Keep searching.
            blocked_wrapper_class = self.venv.getattr_depth_check(already_found, name)

        return blocked_wrapper_class


class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)
