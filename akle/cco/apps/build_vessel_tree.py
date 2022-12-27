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
import numpy as np
import trimesh.creation as tc

from akle.cco import constants
from akle.cco import optimize
from akle.cco.geometry import Point3D, Segment3D
from akle.cco.vessel import Vessel


def _get_coordinates(file_path: Path):
    with open(file_path, newline='') as csvfile:
        reader_object = csv.reader(csvfile, delimiter=',')
        next(reader_object)
        for x, y, z in reader_object:
            yield float(x), float(y), float(z)


def _rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.exp(3) + kmat * kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3)


def main(args: dict[str, Optional[Any]]):
    logger.debug(args)

    coordinates_file = Path(args['<terminals.csv>'])
    points_generator = _get_coordinates(coordinates_file)
    root_inlet = Point3D(*next(points_generator))
    root_outlet = Point3D(*next(points_generator))

    root_vessel = Vessel(inlet=root_inlet,
                         outlet=root_outlet,
                         flow=constants.TERMINAL_FLOW_MM3_PER_SEC,
                         pressure_in=constants.PRESSURE_ENTRY_PASCAL,
                         pressure_out=constants.PRESSURE_OUTLETS_PASCAL)

    vascular_network = {'root': root_vessel,
                        'tree': [root_vessel]}

    for terminal in points_generator:
        logger.debug(f'Processing new terminal... {terminal}')
        optimize.add_terminal(Point3D(*terminal), vascular_network)

    out_dir = Path(args['<output-dir>'])

    for i, vessel in enumerate(vascular_network['tree']):
        output_filename = out_dir / f'vessel_{vessel.index:03d}.ply'
        cylinder = tc.cylinder(radius=vessel.radius,
                               segment=np.vstack([vessel.outlet.coordinates, vessel.inlet.coordinates]))
        cylinder.export(output_filename)


if __name__ == '__main__':
    main(docopt(__doc__, version='build_vessel_tree.py 0.1.0'))
