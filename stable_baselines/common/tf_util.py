import os
import collections
import functools
import multiprocessing
from typing import Set

import numpy as np
import tensorflow.compat.v1 as tf


def is_image(tensor):
    """
    Check if a tensor has the shape of
    a valid image for tensorboard logging.
    Valid image: RGB, RGBD, GrayScale

    :param tensor: (np.ndarray or tf.placeholder)
    :return: (bool)
    """

    return len(tensor.shape) == 3 and tensor.shape[-1] in [1, 3, 4]


def batch_to_seq(tensor_batch, n_batch, n_steps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies

    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_steps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies
    """
    if flat:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps])
    else:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]


def seq_to_batch(tensor_sequence, flat=False):
    """
    Transform a sequence of Tensors, into a batch of Tensors for recurrent policies

    :param tensor_sequence: (TensorFlow Tensor) The input tensor to batch
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) batch of Tensors for recurrent policies
    """
    shape = tensor_sequence[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        n_hidden = tensor_sequence[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
    else:
        return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])


def check_shape(tensors, shapes):
    """
    Verifies the tensors match the given shape, will raise an error if the shapes do not match

    :param tensors: ([TensorFlow Tensor]) The tensors that should be checked
    :param shapes: ([list]) The list of shapes for each tensor
    """
    i = 0
    for (tensor, shape) in zip(tensors, shapes):
        assert tensor.get_shape().as_list() == shape, "id " + str(i) + " shape " + str(tensor.get_shape()) + str(shape)
        i += 1

# ================================================================
# Mathematical utils
# ================================================================


def huber_loss(tensor, delta=1.0):
    """
    Reference: https://en.wikipedia.org/wiki/Huber_loss

    :param tensor: (TensorFlow Tensor) the input value
    :param delta: (float) Huber loss delta value
    :return: (TensorFlow Tensor) Huber loss output
    """
    return tf.where(
        tf.abs(tensor) < delta,
        tf.square(tensor) * 0.5,
        delta * (tf.abs(tensor) - 0.5 * delta)
    )


def sample(logits):
    """
    Creates a sampling Tensor for non deterministic policies
    when using categorical distribution.
    It uses the Gumbel-max trick: http://amid.fish/humble-gumbel

    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The sampled action
    """
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def calc_entropy(logits):
    """
    Calculates the entropy of the output values of the network

    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The Entropy of the output values of the network
    """
    # Compute softmax
    a_0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    z_0 = tf.reduce_sum(exp_a_0, 1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), 1)


def mse(pred, target):
    """
    Returns the Mean squared error between prediction and target

    :param pred: (TensorFlow Tensor) The predicted value
    :param target: (TensorFlow Tensor) The target value
    :return: (TensorFlow Tensor) The Mean squared error between prediction and target
    """
    return tf.reduce_mean(tf.square(pred - target))


def avg_norm(tensor):
    """
    Return an average of the L2 normalization of the batch

    :param tensor: (TensorFlow Tensor) The input tensor
    :return: (TensorFlow Tensor) Average L2 normalization of the batch
    """
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=-1)))


def gradient_add(grad_1, grad_2, param, verbose=0):
    """
    Sum two gradients

    :param grad_1: (TensorFlow Tensor) The first gradient
    :param grad_2: (TensorFlow Tensor) The second gradient
    :param param: (TensorFlow parameters) The trainable parameters
    :param verbose: (int) verbosity level
    :return: (TensorFlow Tensor) the sum of the gradients
    """
    if verbose > 1:
        print([grad_1, grad_2, param.name])
    if grad_1 is None and grad_2 is None:
        return None
    elif grad_1 is None:
        return grad_2
    elif grad_2 is None:
        return grad_1
    else:
        return grad_1 + grad_2


def q_explained_variance(q_pred, q_true):
    """
    Calculates the explained variance of the Q value

    :param q_pred: (TensorFlow Tensor) The predicted Q value
    :param q_true: (TensorFlow Tensor) The expected Q value
    :return: (TensorFlow Tensor) the explained variance of the Q value
    """
    _, var_y = tf.nn.moments(q_true, axes=[0, 1])
    _, var_pred = tf.nn.moments(q_true - q_pred, axes=[0, 1])
    check_shape([var_y, var_pred], [[]] * 2)
    return 1.0 - (var_pred / var_y)


# ================================================================
# Global session
# ================================================================


def make_session(num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


def single_threaded_session(make_default=False, graph=None):
    """
    Returns a session which will only use a single CPU

    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    return make_session(num_cpu=1, make_default=make_default, graph=graph)


def in_session(func):
    """
    Wraps a function so that it is in a TensorFlow Session

    :param func: (function) the function to wrap
    :return: (function)
    """

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        with tf.Session():
            func(*args, **kwargs)

    return newfunc


ALREADY_INITIALIZED = set()  # type: Set[tf.Variable]


def initialize(sess=None):
    """
    Initialize all the uninitialized variables in the global scope.

    :param sess: (TensorFlow Session)
    """
    if sess is None:
        sess = tf.get_default_session()
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


# ================================================================
# Theano-like Function
# ================================================================


def function(inputs, outputs, updates=None, givens=None):
    """
    Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs. Just like a Theano function.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
       >>> x = tf.placeholder(tf.int32, (), name="x")
       >>> y = tf.placeholder(tf.int32, (), name="y")
       >>> z = 3 * x + 2 * y
       >>> lin = function([x, y], z, givens={y: 0})
       >>> with single_threaded_session():
       >>>     initialize()
       >>>     assert lin(2) == 6
       >>>     assert lin(x=3) == 9
       >>>     assert lin(2, 2) == 10

    :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
    :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    :param updates: ([tf.Operation] or tf.Operation)
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
    :param givens: (dict) the values known for the output
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        func = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), func(*args, **kwargs)))
    else:
        func = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: func(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        """
        Theano like function

        :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
        :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
            value will also have the same shape.
        :param updates: ([tf.Operation] or tf.Operation)
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
        :param givens: (dict) the values known for the output
        """
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (isinstance(inpt, tf.Tensor) and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    @classmethod
    def _feed_input(cls, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args, sess=None, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        if sess is None:
            sess = tf.get_default_session()
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = sess.run(self.outputs_update, feed_dict=feed_dict, **kwargs)[:-1]
        return results


# ================================================================
# Flat vectors
# ================================================================


def var_shape(tensor):
    """
    get TensorFlow Tensor shape

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: ([int]) the shape
    """
    out = tensor.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(tensor):
    """
    get TensorFlow Tensor's number of elements

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: (int) the number of elements
    """
    return intprod(var_shape(tensor))


def intprod(tensor):
    """
    calculates the product of all the elements in a list

    :param tensor: ([Number]) the list of elements
    :return: (int) the product truncated
    """
    return int(np.prod(tensor))


def flatgrad(loss, var_list, clip_norm=None):
    """
    calculates the gradient and flattens it

    :param loss: (float) the loss value
    :param var_list: ([TensorFlow Tensor]) the variables
    :param clip_norm: (float) clip the gradients (disabled if None)
    :return: ([TensorFlow Tensor]) flattened gradient
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32, sess=None):
        """
        Set the parameters from a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param dtype: (type) the type for the placeholder
        :param sess: (TensorFlow Session)
        """
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, _var) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(_var, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.operation = tf.group(*assigns)
        self.sess = sess

    def __call__(self, theta):
        if self.sess is None:
            return tf.get_default_session().run(self.operation, feed_dict={self.theta: theta})
        else:
            return self.sess.run(self.operation, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list, sess=None):
        """
        Get the parameters as a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param sess: (TensorFlow Session)
        """
        self.operation = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        if self.sess is None:
            return tf.get_default_session().run(self.operation)
        else:
            return self.sess.run(self.operation)


# ================================================================
# retrieving variables
# ================================================================


def get_trainable_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)


def outer_scope_getter(scope, new_scope=""):
    """
    remove a scope layer for the getter

    :param scope: (str) the layer to remove
    :param new_scope: (str) optional replacement name
    :return: (function (function, str, ``*args``, ``**kwargs``): Tensorflow Tensor)
    """
    def _getter(getter, name, *args, **kwargs):
        name = name.replace(scope + "/", new_scope, 1)
        val = getter(name, *args, **kwargs)
        return val
    return _getter


# ================================================================
# Logging
# ================================================================


def total_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc
