import stable_baselines
import gym
import numpy as np
import warnings

log_dir = "/tmp/gym/"  # support windows?


def testGym(env):
    #env = Monitor(env, log_dir, allow_early_resets=True)

    assert isinstance(env, gym.Env), 'Enviroment is not instance of gym.Env class'

    obs = env.reset()

    assert type(obs) != type(()), 'Your reset function is returning a tuple. It must return the observation array only (not the entire (observation, reward, done, info) tuple like .step(action)'
    assert type(obs) == type(np.array([])), 'The observation you return (in this case from .reset) must be a numpy array'
    assert obs.size != 0, 'The observation .reset returns is an empty numpy array. It should be full. You might want to look into that.'
    assert np.all(obs != np.NaN) == True, 'The observation .reset returns contains NaNs. You might want to look into that.'
    assert np.all(obs != np.Inf) == True, 'The observation .reset returns contains infinity. You might want to look into that.'
    assert np.all(obs != -np.Inf) == True, 'The observation .reset returns contains negative infinity. You might want to look into that.'
    if np.array_equal(obs, np.zeros(obs.shape)):
        warnings.warn('The observation .reset returns is all zeros. You might want to look into that.')
    if not np.all(obs >= 0):
        warnings.warn('The observation .reset returns contains a negative number. If you\'ve made a \'normal\' game, this should not be the case.')

    action = env.action_space.sample()

    out = env.step(action)

    assert type(out) == type(()), 'env.step(action) should return a tuple. If .step(env) ends in \'return observation, reward, done, info\' like it should, the collection of them will be a tuple'

    assert len(out) == 4, 'env.step(action) must return 4 values like this: \'return observation, reward, done, info\'. You probably forgot to add info, if you aren\'t using it just put {} in it\'s place in the return line.'

    observation = out[0]
    reward = out[1]
    done = out[2]
    info = out[3]

    assert observation.shape == obs.shape, 'The deminsions of the observation given by .step and .return are not the same'

    assert obs.size != 0, 'The observation .step returns is an empty numpy array. It should be full. You might want to look into that.'
    assert np.all(observation != np.NaN) == True, 'The observation .step returns contains NaNs. You might want to look into that.'
    assert np.all(observation != np.Inf) == True, 'The observation .step returns contains infinity. You might want to look into that.'
    assert np.all(observation != -np.Inf) == True, 'The observation .step returns contains negative infinity. You might want to look into that.'
    if np.array_equal(obs, np.zeros(obs.shape)):
        warnings.warn('The observation .step returns is all zeros. You might want to look into that.')
    if not np.all(observation >= 0):
        warnings.warn('The observation .step returns contains a negative number. If you\'ve made a \'normal\' game, this should not be the case.')

    assert (type(reward) == type(3)) or (type(reward) == type(8.5)), 'Your reward function must return integer or float type numbers'

    if reward == np.Inf:
        warnings.warn('We took a single random step and your game gave a reward of infinity. You might want to look into that')

    if reward == -np.Inf:
        warnings.warn('We took a single random step and your game gave a reward of negative infinity. You might want to look into that')

    assert type(done) == type(True), 'The done value .step returns must be boolean type (True or False)'

    if done:
        warnings.warn('We took a single random step and your game reached a done state. You might want to look into that')

    assert type(info) == type({}), 'Done returned from .step must return a dictionary type. If you aren\'t using it for anything, just return {}'

# add additional .render and .close tests in a seperate function?
# I don't know what action/state space tests to add
# tests for tests in CI?
# make sure obs are in allowed numpy data types (I don't know which those are or a clean way to do this)
# tests relevant to monitor?

"""
env = gym.make('SpaceInvaders-v0')

testGym(env)
"""