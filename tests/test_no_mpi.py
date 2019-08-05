from importlib import reload

import stable_baselines
from .test_common import _maybe_disable_mpi

def test_no_mpi_no_crash():
    with _maybe_disable_mpi(True):
        reload(stable_baselines)
