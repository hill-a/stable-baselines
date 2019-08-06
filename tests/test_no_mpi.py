from importlib import reload

from types import ModuleType

import stable_baselines
from .test_common import _maybe_disable_mpi


def test_no_mpi_no_crash():
    with _maybe_disable_mpi(True):
        _rreload(stable_baselines)


def _rreload(root_module):
    """Recursively reload a module and all its submodules.

    Graph DFS strategy modified from
    https://stackoverflow.com/questions/15506971/recursive-version-of-reload
    """
    visited = set()
    prefix = root_module.__name__

    def visit(module):
        if module in visited or not module.__name__.startswith(prefix):
            return
        visited.add(module)
        reload(module)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, ModuleType):
                visit(attribute)

    visit(root_module)
