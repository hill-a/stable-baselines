import numpy as np

from stable_baselines.common.schedules import ConstantSchedule, PiecewiseSchedule, LinearSchedule


def test_piecewise_schedule():
    """
    test PiecewiseSchedule
    """
    piecewise_sched = PiecewiseSchedule([(-5, 100), (5, 200), (10, 50), (100, 50), (200, -50)],
                                        outside_value=500)

    assert np.isclose(piecewise_sched.value(-10), 500)
    assert np.isclose(piecewise_sched.value(0), 150)
    assert np.isclose(piecewise_sched.value(5), 200)
    assert np.isclose(piecewise_sched.value(9), 80)
    assert np.isclose(piecewise_sched.value(50), 50)
    assert np.isclose(piecewise_sched.value(80), 50)
    assert np.isclose(piecewise_sched.value(150), 0)
    assert np.isclose(piecewise_sched.value(175), -25)
    assert np.isclose(piecewise_sched.value(201), 500)
    assert np.isclose(piecewise_sched.value(500), 500)

    assert np.isclose(piecewise_sched.value(200 - 1e-10), -50)


def test_constant_schedule():
    """
    test ConstantSchedule
    """
    constant_sched = ConstantSchedule(5)
    for i in range(-100, 100):
        assert np.isclose(constant_sched.value(i), 5)


def test_linear_schedule():
    """
    test LinearSchedule
    """
    linear_sched = LinearSchedule(schedule_timesteps=100, initial_p=0.2, final_p=0.8)
    assert np.isclose(linear_sched.value(50), 0.5)
    assert np.isclose(linear_sched.value(0), 0.2)
    assert np.isclose(linear_sched.value(100), 0.8)
    
    linear_sched = LinearSchedule(schedule_timesteps=100, initial_p=0.8, final_p=0.2)
    assert np.isclose(linear_sched.value(50), 0.5)
    assert np.isclose(linear_sched.value(0), 0.8)
    assert np.isclose(linear_sched.value(100), 0.2)
    
    linear_sched = LinearSchedule(schedule_timesteps=100, initial_p=-0.6, final_p=0.2)
    assert np.isclose(linear_sched.value(50), -0.2)
    assert np.isclose(linear_sched.value(0), -0.6)
    assert np.isclose(linear_sched.value(100), 0.2)
    
    linear_sched = LinearSchedule(schedule_timesteps=100, initial_p=0.2, final_p=-0.6)
    assert np.isclose(linear_sched.value(50), -0.2)
    assert np.isclose(linear_sched.value(0), 0.2)
    assert np.isclose(linear_sched.value(100), -0.6)

