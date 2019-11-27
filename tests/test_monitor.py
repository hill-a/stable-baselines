import uuid
import json
import os

import pandas
import gym

from stable_baselines.bench import Monitor
from stable_baselines.bench.monitor import get_monitor_files, load_results


def test_monitor():
    """
    test the monitor wrapper
    """
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/stable_baselines-test-{}.monitor.csv".format(uuid.uuid4())
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    file_handler = open(mon_file, 'rt')

    firstline = file_handler.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 't_start'}, "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(file_handler, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    file_handler.close()
    os.remove(mon_file)

def test_monitor_load_results(tmp_path):
    """
    test load_results on log files produced by the monitor wrapper
    """
    env1 = gym.make("CartPole-v1")
    env1.seed(0)
    monitor_file1 = os.fspath(tmp_path / "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
    menv1 = Monitor(env1, mon_file1)

    mon_files = get_monitor_files(tmp_path)
    assert len(mon_files) == 1
    assert mon_file1 in mon_files

    menv1.reset()
    for _ in range(1000):
        _, _, done, _ = menv1.step(0)
        if done:
            menv1.reset()

    r1s = load_results(os.fspath(tmp_path)).size
    assert r1s > 0

    env2 = gym.make("CartPole-v1")
    env2.seed(0)
    mon_file2 = os.fspath(tmp_path / "stable_baselines-test-{}.monitor.csv".format(uuid.uuid4()))
    menv2 = Monitor(env2, mon_file2)
    mon_files = get_monitor_files(tmp_path)
    assert len(mon_files) == 2
    assert mon_file1 in mon_files
    assert mon_file2 in mon_files

    menv2.reset()
    for _ in range(1000):
        _, _, done, _ = menv2.step(0)
        if done:
            menv2.reset()

    r2s = load_results(os.fspath(tmp_path)).size

    assert r2s == 2 * r1s

    os.remove(mon_file1)
    os.remove(mon_file2)
