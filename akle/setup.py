# pylint: disable=line-too-long

"""Installs project in your Python environment."""

import os
from pathlib import Path

from setuptools import setup


def find_modules(namespace: str) -> list[str]:
    """Return all modules in namespace."""
    ret = []
    root = Path(__file__).parent / Path(*namespace.split('.'))

    namespace_length = len(namespace.split('.'))
    root_length = len(root.parts)

    for dirname, _, filenames in os.walk(root):
        dir_parts = list(Path(dirname).parts)
        prefix = dir_parts[root_length - namespace_length:]

        modules = []

        for filename in filenames:
            filename = Path(filename)

            if filename.suffix == '.py':
                modules += ['.'.join(prefix + [str(filename.with_suffix(''))])]

        ret += modules

    return ret


setup(
    name='akle-cco',
    description=__doc__,
    install_requires=[
        'docopt',
        'loguru',
        'matplotlib',
        'networkx',
        'numpy',
        'open3d',
        'pandas',
        'scipy',
        'sympy',
        'tqdm',
    ],
    platforms=['Windows', 'Linux'],
    py_modules=find_modules('cco'),
    python_requires='>=3.10',
    version='0.1'
)
