.. _export:


Exporting models
================

After training an agent, you may want to deploy/use it in an other language
or framework, like PyTorch or `tensorflowjs <https://github.com/tensorflow/tfjs>`_.
Stable Baselines does not include tools to export models to other frameworks, but
this document aims to cover parts that are required for exporting along with
more detailed stories from users of Stable Baselines.


Background
----------

In Stable Baselines, the controller is stored inside :ref:`policies <policies>` which convert
observations into actions. Each learning algorithm (e.g. DQN, A2C, SAC) contains
one or more policies, some of which are only used for training. An easy way to find
the policy is to check the code for the ``predict`` function of the agent:
This function should only call one policy with simple arguments.

Policies hold the necessary Tensorflow placeholders and tensors to do the
inference (i.e. predict actions), so it is enough to export these policies
to do inference in an another framework.

.. note::
  Learning algorithms also may contain other Tensorflow placeholders, that are used for training only and are
  not required for inference.


.. warning::
  When using CNN policies, the observation is normalized internally (dividing by 255 to have values in [0, 1])


Export to PyTorch
-----------------

A known working solution is to use :func:`get_parameters <stable_baselines.common.base_class.BaseRLModel.get_parameters>`
function to obtain model parameters, construct the network manually in PyTorch and assign parameters correctly.

.. warning::
  PyTorch and Tensorflow have internal differences with e.g. 2D convolutions (see discussion linked below).


See `discussion #372 <https://github.com/hill-a/stable-baselines/issues/372>`_ for details.


Export to C++
-----------------

Tensorflow, which is the backbone of Stable Baselines, is fundamentally a C/C++ library despite being most commonly accessed
through the Python frontend layer. This design choice means that the models created at Python level should generally be
fully compliant with the respective C++ version of Tensorflow.

.. warning::
   It is advisable not to mix-and-match different versions of Tensorflow libraries, particularly in terms of the state.
   Moving computational graphs is generally more forgiving. As a matter of fact, mentioned below `PPO_CPP <https://github.com/Antymon/ppo_cpp>`_ project uses
   graphs generated with Python Tensorflow 1.x in C++ Tensorflow 2 version.

Stable Baselines comes very handily when hoping to migrate a computational graph and/or a state (weights) as
the existing algorithms define most of the necessary computations for you so you don't need to recreate the core of the algorithms again.
This is exactly the idea that has been used in the `PPO_CPP <https://github.com/Antymon/ppo_cpp>`_ project, which executes the training at the C++ level for the sake of
computational efficiency. The graphs are exported from Stable Baselines' PPO2 implementation through ``tf.train.export_meta_graph``
function. Alternatively, and perhaps more commonly, you could use the C++ layer only for inference. That could be useful
as a deployment step of server backends or optimization for more limited devices.

.. warning::
   As a word of caution, C++-level APIs are more imperative than their Python counterparts or more plainly speaking: cruder.
   This is particularly apparent in Tensorflow 2.0 where the declarativeness of Autograph exists only at Python level. The
   C++ counterpart still operates on Session objects' use, which are known from earlier versions of Tensorflow. In our use case,
   availability of graphs utilized by Session depends on the use of ``tf.function`` decorators. However, as of November 2019, Stable Baselines still
   uses Tensorflow 1.x in the main version which is slightly easier to use in the context of the C++ portability.


Export to tensorflowjs / tfjs
-----------------------------

Can be done via Tensorflow's `simple_save <https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_ function
and `tensorflowjs_converter <https://www.tensorflow.org/js/tutorials/conversion/import_saved_model>`_.

See `discussion #474 <https://github.com/hill-a/stable-baselines/issues/474>`_ for details.


Export to Java
---------------

Can be done via Tensorflow's `simple_save <https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_ function.

See `this discussion <https://github.com/hill-a/stable-baselines/issues/329>`_ for details.


Manual export
-------------

You can also manually export required parameters (weights) and construct the
network in your desired framework, as done with the PyTorch example above.

You can access parameters of the model via agents'
:func:`get_parameters <stable_baselines.common.base_class.BaseRLModel.get_parameters>`
function. If you use default policies, you can find the architecture of the networks in
source for :ref:`policies <policies>`. Otherwise, for DQN/SAC/DDPG or TD3 you need to check the `policies.py` file located
in their respective folders.
