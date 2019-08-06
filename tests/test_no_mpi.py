from importlib import reload

from types import ModuleType

import stable_baselines
from .test_common import _maybe_disable_mpi


def test_no_mpi_no_crash():
    with _maybe_disable_mpi(True):
        rreload(stable_baselines)


def rreload(module):
    """Recursively reload a module and all its submodules.

    Graph DFS strategy modified from
    https://stackoverflow.com/questions/15506971/recursive-version-of-reload
    """
    visited = set()
    def visit(m):
        if m in visited:
            return
        visited.add(m)
        reload(module)
        for attribute_name in dir(m):
            attribute = getattr(m, attribute_name)
            if type(attribute) is ModuleType:
                visit(attribute)
    visit(module)
