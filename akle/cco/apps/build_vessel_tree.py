"""Performs test of age prediction model on input data folders.
Usage:
  build_vessel_tree.py [options] <terminals.csv> <output-dir>
  build_vessel_tree.py -h | --help
  build_vessel_tree.py --version
Arguments:
  <terminals.csv>	        Path to csv file with terminals coordinates.
  <output-dir>              Path where to store results.
Options:
  -h --help		            Show this screen.
  --version		            Show version.
"""
import csv
from pathlib import Path
from typing import Any, Optional

from docopt import docopt
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from sympy import Point3D
from tqdm import tqdm

from akle.cco import constants
from akle.cco import optimize
from akle.cco.vessel import Vessel


def get_coordinates(file_path: Path):
    with open(file_path, newline='') as csvfile:
        reader_object = csv.reader(csvfile, delimiter=',')
        next(reader_object)
        for x, y, z in reader_object:
            yield float(x), float(y), float(z)


def main(args: dict[str, Optional[Any]]):
    logger.debug(args)

    coordinates_file = Path(args['<terminals.csv>'])
    root_inlet = Point3D(get_coordinates(coordinates_file))
    root_outlet = Point3D(get_coordinates(coordinates_file))

    root_vessel = Vessel(inlet=root_inlet,
                         outlet=root_outlet,
                         flow=constants.TERMINAL_FLOW_MM3_PER_SEC,
                         pressure_in=constants.PRESSURE_ENTRY_PASCAL,
                         pressure_out=constants.PRESSURE_OUTLETS_PASCAL)

    vascular_network = [root_vessel]

    for terminal in get_coordinates(coordinates_file):

        pass


if __name__ == '__main__':
    main(docopt(__doc__, version='build_vessel_tree.py 0.1.0'))
    plt.close('all')
