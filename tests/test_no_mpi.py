import sys

from .test_common import _maybe_disable_mpi


def test_no_mpi_no_crash():
    with _maybe_disable_mpi(True):
        sb_modules = [name for name in sys.modules.keys()
                      if name.startswith('stable_baselines')]
        for name in sb_modules:
            del sys.modules[name]
        import stable_baselines
        del stable_baselines  # appease Codacy