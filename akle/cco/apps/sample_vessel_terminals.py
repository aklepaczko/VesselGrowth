"""Randomly samples perfusion volume to generate set of vessel tree terminal points.

Usage:
  sample_vessel_terminals.py [options] --radius <value> <terminals.csv>
  sample_vessel_terminals.py -h | --help
  sample_vessel_terminals.py --version

Arguments:
  <terminals.csv>       Path to output csv file with terminals coordinates.

Options:
  --radius <value>      Radius of the spherical perfusion volume in mm.
  --count <value>       Number of terminal points [default: 50]
  -h --help		        Show this screen.
  --version		        Show version.
"""
from pathlib import Path
from typing import Any, Optional

from docopt import docopt
from loguru import logger
import numpy as np
import pandas as pd


def main(args: dict[str, Optional[Any]]):
    logger.debug(args)

    num_terminals = int(args['--count'])
    perfusion_volume_radius = float(args['--radius'])

    radii = np.random.random_sample(num_terminals) * perfusion_volume_radius
    azimuths = np.random.random_sample(num_terminals) * 2 * np.pi
    elevations = np.random.rand(num_terminals) * np.pi

    xx = radii * np.sin(elevations) * np.cos(azimuths)
    yy = radii * np.sin(elevations) * np.sin(azimuths)
    zz = radii * np.cos(elevations)

    terminals = np.concatenate([xx[:, None], yy[:, None], zz[:, None]], axis=1)
    terminals = np.vstack([np.array([perfusion_volume_radius, 0, 0]), terminals])
    df = pd.DataFrame(data=terminals,
                      columns=['x', 'y', 'z'])
    output_filename = args['<terminals.csv>']
    df.to_csv(output_filename, sep=',', index=False)


if __name__ == '__main__':
    main(docopt(__doc__, version='sample_vessel_terminals.py 0.1.0'))
